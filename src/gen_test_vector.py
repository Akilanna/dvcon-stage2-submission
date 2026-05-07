#!/usr/bin/env python3
"""
scripts/gen_test_vector.py

Generates a single 731-D fixed-point test vector for ranking_mlp_top.

Output:
  test_vector.bin  --  731 × int16_t, little-endian, row-major
                       Each value is ap_fixed<16,6> (10 fractional bits)

Usage:
  python scripts/gen_test_vector.py              # random seed=42, range [-2,2]
  python scripts/gen_test_vector.py --seed 7     # different seed
  python scripts/gen_test_vector.py --ramp       # deterministic ramp [-1,1]
  python scripts/gen_test_vector.py --out /tmp/vec.bin
"""

import argparse
import os
import struct
import sys

# ---------------------------------------------------------------------------
# Constants (must match top.h)
# ---------------------------------------------------------------------------
IN_DIM       = 731
FRAC_BITS    = 10
FP_SCALE     = 1 << FRAC_BITS      # 1024
FP_MAX_INT   =  (1 << 15) - 1      #  32767
FP_MIN_INT   = -(1 << 15)          # -32768
FLOAT_MAX    =  FP_MAX_INT / FP_SCALE
FLOAT_MIN    =  FP_MIN_INT / FP_SCALE


def float_to_fp(x: float) -> int:
    """Saturating quantise: float → ap_fixed<16,6> int16."""
    if x > FLOAT_MAX: x = FLOAT_MAX
    if x < FLOAT_MIN: x = FLOAT_MIN
    raw = round(x * FP_SCALE)
    # clamp after rounding
    return max(FP_MIN_INT, min(FP_MAX_INT, raw))


def generate_random(seed: int) -> list:
    """731 uniform random floats in [-2.0, +2.0]."""
    import random
    rng = random.Random(seed)
    return [rng.uniform(-2.0, 2.0) for _ in range(IN_DIM)]


def generate_ramp() -> list:
    """Deterministic ramp [-1.0, +1.0] matching ranking_test.c test vector."""
    return [(float(j % 64 - 32) / 32.0) for j in range(IN_DIM)]


def write_bin(floats: list, path: str) -> None:
    raw = [float_to_fp(v) for v in floats]
    with open(path, "wb") as f:
        for v in raw:
            f.write(struct.pack("<h", v))   # little-endian signed int16
    print(f"[WRITE] {path}  ({IN_DIM} values × 2 bytes = {IN_DIM*2} bytes)")


def print_stats(floats: list) -> None:
    raw = [float_to_fp(v) for v in floats]
    recovered = [r / FP_SCALE for r in raw]
    errors = [abs(f - r) for f, r in zip(floats, recovered)]
    clamped = sum(1 for v in floats if v > FLOAT_MAX or v < FLOAT_MIN)

    print(f"  Dimension    : {IN_DIM}")
    print(f"  Float range  : [{min(floats):+.4f}, {max(floats):+.4f}]")
    print(f"  Clamped      : {clamped}/{IN_DIM}")
    print(f"  Max |error|  : {max(errors):.6f}  (≤{1.0/FP_SCALE:.6f} expected)")
    print(f"  Preview [0]  : float={floats[0]:+.4f}  fp={raw[0]}  back={recovered[0]:+.4f}")
    print(f"  Preview [365]: float={floats[365]:+.4f}  fp={raw[365]}  back={recovered[365]:+.4f}")
    print(f"  Preview [730]: float={floats[730]:+.4f}  fp={raw[730]}  back={recovered[730]:+.4f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a 731-D ap_fixed<16,6> test vector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed",  type=int, default=42,
                   help="Random seed")
    p.add_argument("--ramp",  action="store_true",
                   help="Use deterministic ramp instead of random")
    p.add_argument("--out",   type=str, default="test_vector.bin",
                   help="Output binary file path")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress statistics")
    return p.parse_args()


def main():
    args = parse_args()

    if args.ramp:
        floats = generate_ramp()
        print(f"Mode: deterministic ramp  (matches ranking_test.c built-in vector)")
    else:
        floats = generate_random(args.seed)
        print(f"Mode: random  seed={args.seed}  range=[-2.0, +2.0]")

    if not args.quiet:
        print_stats(floats)
        print()

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    write_bin(floats, args.out)

    print()
    print("=== DMA parameters for this vector ===")
    print(f"  TX bytes = {IN_DIM} × 2 = {IN_DIM * 2}  (1 object)")
    print(f"  RX bytes = 1 × 2 = 2                   (1 score)")
    print(f"  Load via JTAG: xsct> dow -data {args.out} 0x<tx_buf_addr>")


if __name__ == "__main__":
    main()
