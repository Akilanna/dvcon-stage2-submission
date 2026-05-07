#!/usr/bin/env python3
"""
generate_vectors.py

Generates synthetic 731-D feature vectors for the ranking_mlp_top accelerator.

Output files:
  input_vectors.bin            -- binary int16_t, row-major [N, 731]
  input_vectors_preview.csv    -- human-readable float preview (first 5 objects)
  test_config.h                -- C header with dimensions for the driver

Fixed-point format matches top.h:
  input_t = ap_fixed<16,6>
    16 total bits, 6 integer bits (including sign sign bit)
    fractional bits : 10
    range           : [-32.0, +31.999023)
    step            : 2^-10 = 1/1024 ≈ 0.000977
    int16_t raw     = round(float * 1024)

Feature vector layout (must match models/ranker/ranking_mlp.py):
  [  0..127] object features   (128-D YOLO backbone output, normalised)
  [128..255] scene descriptor  (128-D pooled scene feature)
  [256..639] task embedding    (384-D CLIP text embedding, unit-normalised)
  [640..730] class one-hot     ( 91-D COCO class vector)
  Total: 128 + 128 + 384 + 91 = 731

Usage:
  python scripts/generate_vectors.py               # 14 objects, seed=42
  python scripts/generate_vectors.py -n 100 -s 7   # 100 objects, seed=7
  python scripts/generate_vectors.py -n 1 --no-csv --out-dir /tmp
"""

import argparse
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Network / fixed-point constants (must match top.h)
# ---------------------------------------------------------------------------
IN_DIM      = 731
FRAC_BITS   = 10
FP_SCALE    = 1 << FRAC_BITS        # 1024
FP_MAX      =  (1 << 15) - 1        #  32767
FP_MIN      = -(1 << 15)            # -32768
FLOAT_MAX   =  FP_MAX / FP_SCALE    #  31.999023...
FLOAT_MIN   =  FP_MIN / FP_SCALE    # -32.0

# Feature segment boundaries
SEG_OBJECT = (0,   128)
SEG_SCENE  = (128, 256)
SEG_TASK   = (256, 640)
SEG_CLASS  = (640, 731)
NUM_CLASSES = SEG_CLASS[1] - SEG_CLASS[0]   # 91

assert SEG_CLASS[1] == IN_DIM, "Segment boundaries must sum to IN_DIM"


# ---------------------------------------------------------------------------
# Fixed-point conversion
# ---------------------------------------------------------------------------

def float_to_fp16(x: np.ndarray) -> np.ndarray:
    """
    Quantise float32 array to ap_fixed<16,6> int16 representation.
    Clamps values outside [-32.0, +31.999] before conversion.
    """
    clipped = np.clip(x, FLOAT_MIN, FLOAT_MAX)
    raw = np.round(clipped * FP_SCALE).astype(np.int16)
    return raw


def fp16_to_float(x: np.ndarray) -> np.ndarray:
    """Inverse: recover float from fixed-point int16."""
    return x.astype(np.float32) / FP_SCALE


# ---------------------------------------------------------------------------
# Synthetic vector generation
# ---------------------------------------------------------------------------

def generate_objects(n: int, seed: int = 42) -> np.ndarray:
    """
    Generate n random feature vectors (float32, shape [n, IN_DIM]).

    Each segment uses a distribution representative of real data:
      - object / scene features : N(0, 0.5), clipped to [-2, 2]
      - task embedding          : unit-sphere normalised N(0,1), scaled ×2
      - class one-hot           : exactly one '1' per object
    """
    rng = np.random.default_rng(seed)
    vecs = np.zeros((n, IN_DIM), dtype=np.float32)

    # Object features
    s, e = SEG_OBJECT
    vecs[:, s:e] = np.clip(rng.normal(0.0, 0.5, (n, e - s)), -2.0, 2.0)

    # Scene descriptor
    s, e = SEG_SCENE
    vecs[:, s:e] = np.clip(rng.normal(0.0, 0.5, (n, e - s)), -2.0, 2.0)

    # Task embedding — unit-sphere normalised, then scaled to use dynamic range
    s, e = SEG_TASK
    raw  = rng.normal(0.0, 1.0, (n, e - s)).astype(np.float32)
    norm = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
    vecs[:, s:e] = np.clip((raw / norm) * 2.0, -2.0, 2.0)

    # Class one-hot (COCO: 91 classes)
    s, e = SEG_CLASS
    classes = rng.integers(0, NUM_CLASSES, size=n)
    for i, c in enumerate(classes):
        vecs[i, s + c] = 1.0

    return vecs


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_binary(vecs: np.ndarray, path: str) -> None:
    """Write quantised int16 row-major binary file."""
    q = float_to_fp16(vecs)
    q.tofile(path)
    print(f"[WRITE] {path}  ({vecs.shape[0]} objects × {IN_DIM} features, "
          f"{q.nbytes} bytes, little-endian int16)")


def write_csv(vecs: np.ndarray, path: str, n_rows: int = 5) -> None:
    """Write human-readable CSV of first n_rows objects (as recovered floats)."""
    q        = float_to_fp16(vecs)
    recovered = fp16_to_float(q)
    with open(path, "w") as f:
        f.write("# Fixed-point recovered floats — ap_fixed<16,6>\n")
        f.write("# obj_index," + ",".join(f"feat_{j}" for j in range(IN_DIM)) + "\n")
        for i in range(min(n_rows, len(vecs))):
            row = [str(i)] + [f"{v:.6f}" for v in recovered[i]]
            f.write(",".join(row) + "\n")
    print(f"[WRITE] {path}  (preview: {min(n_rows, len(vecs))} objects)")


def write_header(n: int, path: str) -> None:
    """Write C header so the driver knows how many objects to expect."""
    with open(path, "w") as f:
        f.write("/* Auto-generated by scripts/generate_vectors.py — do not edit */\n")
        f.write("#ifndef TEST_CONFIG_H\n")
        f.write("#define TEST_CONFIG_H\n\n")
        f.write(f"#define TEST_N_OBJECTS    {n}\n")
        f.write(f"#define TEST_IN_DIM       {IN_DIM}\n")
        f.write(f"#define TEST_FP_FRAC_BITS {FRAC_BITS}\n")
        f.write(f'#define TEST_VECTORS_FILE "input_vectors.bin"\n')
        f.write(f"/* TX bytes = TEST_N_OBJECTS * TEST_IN_DIM * 2 = "
                f"{n * IN_DIM * 2} */\n")
        f.write(f"/* RX bytes = TEST_N_OBJECTS * 2            = "
                f"{n * 2} */\n")
        f.write("\n#endif /* TEST_CONFIG_H */\n")
    print(f"[WRITE] {path}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_stats(vecs: np.ndarray) -> None:
    q         = float_to_fp16(vecs)
    recovered = fp16_to_float(q)
    err       = np.abs(vecs - recovered)
    clamped   = ((vecs > FLOAT_MAX) | (vecs < FLOAT_MIN)).sum()
    total     = vecs.size

    print("\n--- Quantisation Statistics ---")
    print(f"  Vectors       : {vecs.shape[0]} × {IN_DIM}")
    print(f"  Input range   : [{vecs.min():+.4f}, {vecs.max():+.4f}]")
    print(f"  Clamped values: {clamped}/{total}  "
          f"({100.0 * clamped / total:.3f}%)")
    print(f"  Max abs error : {err.max():.6f}  (≤ {1.0/FP_SCALE:.6f} expected)")
    print(f"  Mean abs error: {err.mean():.6f}")
    print()

    # Per-segment stats
    segs = [("object", *SEG_OBJECT),
            ("scene",  *SEG_SCENE),
            ("task",   *SEG_TASK),
            ("class",  *SEG_CLASS)]
    print("  Per-segment max |value|:")
    for name, s, e in segs:
        print(f"    {name:10s} [{s:3d}:{e:3d}]  "
              f"max={np.abs(vecs[:, s:e]).max():.4f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate fixed-point test vectors for ranking_mlp_top",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-n", "--n-objects", type=int,  default=14,
                   help="Number of objects to generate")
    p.add_argument("-s", "--seed",      type=int,  default=42,
                   help="Random seed for reproducibility")
    p.add_argument("-o", "--out-dir",   type=str,  default="fpga/driver",
                   help="Output directory for generated files")
    p.add_argument("--no-csv",          action="store_true",
                   help="Skip CSV preview file")
    p.add_argument("--no-header",       action="store_true",
                   help="Skip C header file")
    p.add_argument("--quiet",           action="store_true",
                   help="Suppress statistics output")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Generating {args.n_objects} object vectors "
          f"(seed={args.seed}, out={args.out_dir})")

    vecs = generate_objects(args.n_objects, seed=args.seed)

    if not args.quiet:
        print_stats(vecs)

    write_binary(vecs, os.path.join(args.out_dir, "input_vectors.bin"))

    if not args.no_csv:
        write_csv(vecs, os.path.join(args.out_dir, "input_vectors_preview.csv"))

    if not args.no_header:
        write_header(args.n_objects,
                     os.path.join(args.out_dir, "test_config.h"))

    print()
    print("=== Driver command ===")
    print(f"  TX: {args.n_objects} × {IN_DIM} × 2 = "
          f"{args.n_objects * IN_DIM * 2} bytes")
    print(f"  RX: {args.n_objects} × 2             = "
          f"{args.n_objects * 2} bytes")
    print(f"  Expected throughput @ 200 MHz: "
          f"{args.n_objects * 731 / 200e6 * 1e6:.1f} µs")


if __name__ == "__main__":
    main()
