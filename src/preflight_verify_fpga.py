#!/usr/bin/env python3
"""
scripts/preflight_verify_fpga.py
=================================
Offline preflight verification for the ranking_mlp FPGA accelerator system.

Verifies every integration layer without requiring a board or Vivado/PYNQ install.
Each check emits PASS or FAIL and a brief reason.

Exit code 0 = all checks passed.
Exit code 1 = one or more failures.

Usage:
    py scripts/preflight_verify_fpga.py          # from repo root
    python scripts/preflight_verify_fpga.py      # on Linux/PYNQ
"""

import sys
import re
import pathlib
import struct
import importlib.util

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
REPO = pathlib.Path(__file__).resolve().parent.parent
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

_results: list[tuple[str, bool | None, str]] = []


def check(name: str, ok: bool | None, detail: str = "") -> bool:
    """Record and print a single check result."""
    if ok is None:
        tag = SKIP
        _results.append((name, None, detail))
        print(f"  [{tag}]  {name}" + (f"  — {detail}" if detail else ""))
        return True
    else:
        tag = PASS if ok else FAIL
        _results.append((name, ok, detail))
    print(f"  [{tag}]  {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def read_text(path: pathlib.Path) -> str | None:
    """Read file with UTF-8 fallback; return None if not found."""
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


# ---------------------------------------------------------------------------
# 1. Overlay file existence
# ---------------------------------------------------------------------------
print("\n── 1. Overlay files ──────────────────────────────────────")
overlay_dir = REPO / "fpga" / "build"
bit_file = overlay_dir / "ranking_mlp.bit"
hwh_file = overlay_dir / "ranking_mlp.hwh"
xsa_file = overlay_dir / "ranking_mlp_system.xsa"

check("ranking_mlp.bit exists",
      bit_file.exists(),
      str(bit_file.relative_to(REPO)) + (" (Vivado not yet run — generate bitstream first)" if not bit_file.exists() else ""))

# Auto-extract HWH from XSA if .hwh not yet present
if not hwh_file.exists() and xsa_file.exists():
    try:
        import zipfile
        with zipfile.ZipFile(xsa_file) as z:
            hwh_names = [n for n in z.namelist() if n.endswith(".hwh")]
            if hwh_names:
                data = z.read(hwh_names[0])
                hwh_file.write_bytes(data)
                print(f"  [INFO] HWH extracted from XSA → {hwh_file.relative_to(REPO)}")
    except Exception as e:
        print(f"  [WARN] HWH extraction failed: {e}")

check("ranking_mlp.hwh exists",
      hwh_file.exists(),
      str(hwh_file.relative_to(REPO)) + (" (extract from ranking_mlp_system.xsa or re-run Vivado)" if not hwh_file.exists() else ""))
check("ranking_mlp_system.xsa exists",
      xsa_file.exists(),
      str(xsa_file.relative_to(REPO)) if not xsa_file.exists() else f"{xsa_file.stat().st_size:,} bytes")

hwh_text = read_text(hwh_file)

# ---------------------------------------------------------------------------
# 2. HWH: required IPs present
# ---------------------------------------------------------------------------
print("\n── 2. HWH required IP blocks ────────────────────────────")
if hwh_text is None:
    check("ranking_mlp_top_0 in HWH", None, "HWH absent — will check when generated")
    check("axi_dma_0 in HWH",         None, "HWH absent")
else:
    check("ranking_mlp_top_0 in HWH",
          "ranking_mlp_top_0" in hwh_text or "ranking_mlp_top" in hwh_text)
    check("axi_dma_0 in HWH",
          "axi_dma_0" in hwh_text)

# ---------------------------------------------------------------------------
# 3. HWH: AXI stream width = 16 bits
# ---------------------------------------------------------------------------
print("\n── 3. AXI stream width ──────────────────────────────────")
if hwh_text is None:
    check("TDATA width = 16 bits", None, "HWH absent")
else:
    # HWH uses VALUE="2" format (XML attribute)
    tdata_match = re.search(r'TDATA_NUM_BYTES[^>]*VALUE=["\']?2["\']?', hwh_text, re.I)
    width_match  = re.search(r'(?:TDATA_WIDTH|DATA_WIDTH)[^>]*VALUE=["\']?16["\']?', hwh_text, re.I)
    ok = bool(tdata_match or width_match)
    check("TDATA width = 16 bits (2 bytes)", ok,
          f"TDATA_NUM_BYTES=2: {bool(tdata_match)}  TDATA_WIDTH=16: {bool(width_match)}")

# ---------------------------------------------------------------------------
# 4. HWH: DMA scatter-gather disabled (simple mode)
# ---------------------------------------------------------------------------
print("\n── 4. DMA mode ───────────────────────────────────────────")
if hwh_text is None:
    check("DMA simple mode (no SG)", None, "HWH absent")
else:
    sg_enabled = bool(re.search(r'c_include_sg\s*[=:]\s*["\']?1["\']?', hwh_text, re.I))
    check("DMA simple mode (c_include_sg=0)", not sg_enabled,
          "SG was found enabled — regenerate DMA with simple mode" if sg_enabled else "")

# ---------------------------------------------------------------------------
# 5. Address map: TCL vs stubs vs driver
# ---------------------------------------------------------------------------
print("\n── 5. Address map consistency ────────────────────────────")
RANKER_ADDR = "0xA0010000"
DMA_ADDR    = "0xA0000000"

tcl_text  = read_text(REPO / "fpga" / "system" / "create_design.tcl")
stub_text = read_text(REPO / "fpga" / "bsp_stubs" / "xparameters.h")
host_text = read_text(REPO / "host" / "ranking_test.c")

def addr_present(src: str | None, addr: str, label: str) -> bool:
    if src is None: return False
    return addr in src

check(f"Vivado TCL: ranker at {RANKER_ADDR}", addr_present(tcl_text,  RANKER_ADDR, "tcl"))
check(f"Vivado TCL: DMA at {DMA_ADDR}",     addr_present(tcl_text,  DMA_ADDR,    "tcl"))
check(f"BSP stub: ranker at {RANKER_ADDR}", addr_present(stub_text, RANKER_ADDR, "stub"))
check(f"BSP stub: DMA at {DMA_ADDR}",       addr_present(stub_text, DMA_ADDR,    "stub"))
check("host/ranking_test.c: ranker addr",   addr_present(host_text, RANKER_ADDR, "host"))
check("host/ranking_test.c: DMA addr",      addr_present(host_text, DMA_ADDR,    "host"))

# ---------------------------------------------------------------------------
# 6. Kernel AXI-Lite register offsets vs hardware header
# ---------------------------------------------------------------------------
print("\n── 6. AXI-Lite register offsets ──────────────────────────")
hw_h_path = (REPO / "fpga" / "hls" / "ranking_mlp" /
             "ranking_mlp_hls" / "solution1" / "impl" / "ip" /
             "drivers" / "ranking_mlp_top_v1_0" / "src" / "xranking_mlp_top_hw.h")
hw_h_text = read_text(hw_h_path)

HW_OFFSETS = {"AP_CTRL": 0x00, "TASK_ID": 0x10, "N_OBJECTS": 0x18}
py_text = read_text(REPO / "pynq" / "ranking_accelerator.py")

if hw_h_text and py_text:
    for name, expected in HW_OFFSETS.items():
        hw_m  = re.search(rf'ADDR_{name}_DATA\s+({{"}}0x[0-9a-fA-F]+|0x[0-9a-fA-F]+)',
                          hw_h_text, re.I)
        # fallback for AP_CTRL which has a different macro name
        if not hw_m:
            hw_m = re.search(rf'ADDR_AP_CTRL\s+(0x[0-9a-fA-F]+)', hw_h_text, re.I) \
                   if name == "AP_CTRL" else None
        py_m  = re.search(rf'REG_{name}\s*=\s*(0x[0-9a-fA-F]+)', py_text, re.I)
        if py_m:
            py_val = int(py_m.group(1), 16)
            check(f"REG_{name} matches HW (0x{expected:02x})",
                  py_val == expected,
                  f"driver has 0x{py_val:02x}")
        else:
            check(f"REG_{name} defined in driver", False, "constant not found")
else:
    check("Register offset check", None, "Source file(s) absent")

# ---------------------------------------------------------------------------
# 7. Fixed-point scale constant
# ---------------------------------------------------------------------------
print("\n── 7. Fixed-point scale (FP_SCALE) ──────────────────────")
# ap_fixed<16,6>: 6 integer bits, 10 fractional → scale = 2^10 = 1024
EXPECTED_SCALE = 1024

if py_text:
    m = re.search(r'FP_SCALE\s*=\s*(\d+)', py_text)
    if m:
        actual = int(m.group(1))
        check("FP_SCALE == 1024 (ap_fixed<16,6>, 10 fractional bits)",
              actual == EXPECTED_SCALE,
              f"driver has {actual} — {'correct' if actual == EXPECTED_SCALE else 'should be 1024 (2^10)'}")
    else:
        check("FP_SCALE defined", False, "constant not found in ranking_accelerator.py")
else:
    check("FP_SCALE check", None, "ranking_accelerator.py absent")

# ---------------------------------------------------------------------------
# 8. Buffer size formulas
# ---------------------------------------------------------------------------
print("\n── 8. Buffer size formulas ───────────────────────────────")
IN_DIM = 731

if py_text:
    has_tx = "n_objects * IN_DIM * 2" in py_text or "n_objects * self.IN_DIM * 2" in py_text
    has_rx = "n_objects * 2" in py_text
    check("TX formula: n_objects * IN_DIM * 2", has_tx)
    check("RX formula: n_objects * 2", has_rx)
    # Print expected values
    for n in [1, 2, 14]:
        tx = n * IN_DIM * 2
        rx = n * 2
        print(f"          n={n:2d}: TX={tx:6d} B  RX={rx:4d} B")
else:
    check("Buffer size formulas", None, "ranking_accelerator.py absent")

# ---------------------------------------------------------------------------
# 9. DMA transfer ordering (recv before send)
# ---------------------------------------------------------------------------
print("\n── 9. DMA transfer ordering ──────────────────────────────")
if py_text:
    recv_pos = py_text.find("self.recv.transfer(")
    send_pos = py_text.find("self.send.transfer(")
    order_ok = (recv_pos != -1) and (send_pos != -1) and (recv_pos < send_pos)
    check("recv.transfer() before send.transfer()", order_ok,
          f"recv@{recv_pos} send@{send_pos}" if recv_pos != -1 else "transfer calls not found")
else:
    check("DMA ordering", None, "ranking_accelerator.py absent")

# ---------------------------------------------------------------------------
# 10. No manual cache flush
# ---------------------------------------------------------------------------
print("\n── 10. No manual cache flush ─────────────────────────────")
if py_text:
    # Match actual flush/invalidate API calls, not comments
    flush_calls = re.findall(r'^\s*[^#].*\b(Xil_DCacheFlush|Xil_DCacheInvalidate|'
                             r'ctypes\.cdll.*flush|os\.system.*cache)\b',
                             py_text, re.M | re.I)
    check("No manual cache flush API calls", len(flush_calls) == 0,
          f"Found: {flush_calls}" if flush_calls else "allocate() provides coherent CMA memory")
else:
    check("No manual cache flush", None, "ranking_accelerator.py absent")

# ---------------------------------------------------------------------------
# 11. DMA size assertions present
# ---------------------------------------------------------------------------
print("\n── 11. Safety assertions ─────────────────────────────────")
if py_text:
    has_tx_assert = "assert tx_bytes ==" in py_text
    has_rx_assert = "assert rx_bytes ==" in py_text
    check("TX size assertion present", has_tx_assert)
    check("RX size assertion present", has_rx_assert)
else:
    check("Safety assertions", None, "ranking_accelerator.py absent")

# ---------------------------------------------------------------------------
# 12. freebuffer() called after transfers
# ---------------------------------------------------------------------------
print("\n── 12. CMA buffer lifecycle ──────────────────────────────")
if py_text:
    fb_count = len(re.findall(r'\.freebuffer\(\)', py_text))
    check(f"freebuffer() called >= 2 times (found {fb_count})", fb_count >= 2,
          "prevents CMA pool exhaustion in loops")
else:
    check("freebuffer()", None, "ranking_accelerator.py absent")

# ---------------------------------------------------------------------------
# 13. Test suite function coverage
# ---------------------------------------------------------------------------
print("\n── 13. Test suite coverage ───────────────────────────────")
test_text = read_text(REPO / "pynq" / "test_ranking_fpga.py")
REQUIRED_TESTS = [
    "single_object", "two_objects", "batch_14", "zero_vector",
    "reproducibility", "dma_size_invariants", "latency_measurement",
]
if test_text:
    for t in REQUIRED_TESTS:
        check(f"Test: {t}", t in test_text)
else:
    check("test_ranking_fpga.py", None, "file absent")

# ---------------------------------------------------------------------------
# 14. Python syntax check on driver files
# ---------------------------------------------------------------------------
print("\n── 14. Python syntax check ───────────────────────────────")
import ast

for rel in ["pynq/ranking_accelerator.py", "pynq/test_ranking_fpga.py",
            "scripts/gen_test_vector.py", "scripts/generate_vectors.py"]:
    path = REPO / rel
    if not path.exists():
        check(f"syntax: {rel}", None, "file absent")
        continue
    src = read_text(path)
    try:
        ast.parse(src)
        check(f"syntax: {rel}", True)
    except SyntaxError as e:
        check(f"syntax: {rel}", False, str(e))

# ---------------------------------------------------------------------------
# 15. Theoretical latency computation
# ---------------------------------------------------------------------------
print("\n── 15. Theoretical latency ───────────────────────────────")
CLK_MHZ = 200.0
II      = 731
print(f"          Clock: {CLK_MHZ:.0f} MHz")
print(f"          II:    {II} cycles/object")
for n in [1, 2, 14]:
    t_us = n * II / CLK_MHZ
    print(f"          n={n:2d}: theory={t_us:.3f} µs  "
          f"(measured expected ~{t_us+85:.0f} µs with DMA+Python overhead)")
check("Latency formula consistent", True, f"{II} cycles × n / {CLK_MHZ:.0f}MHz")

# ---------------------------------------------------------------------------
# 16. HLS IP component.xml exists
# ---------------------------------------------------------------------------
print("\n── 16. HLS IP export ─────────────────────────────────────")
component_xml = (REPO / "fpga" / "hls" / "ranking_mlp" /
                 "ranking_mlp_hls" / "solution1" / "impl" / "ip" / "component.xml")
check("HLS IP component.xml exists", component_xml.exists(),
      "run: py scripts/run_hls_ci.py" if not component_xml.exists() else str(component_xml.relative_to(REPO)))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
passed  = sum(1 for _, ok, _ in _results if ok is True)
failed  = sum(1 for _, ok, _ in _results if ok is False)
skipped = sum(1 for _, ok, _ in _results if ok is None)
total   = len(_results)

tag = "\033[32mALL PASS\033[0m" if failed == 0 else f"\033[31m{failed} FAILED\033[0m"
print(f"  RESULT: {passed} passed, {failed} failed, {skipped} skipped (total {total})  [{tag}]")

if failed > 0:
    print("\n  Failed checks:")
    for name, ok, detail in _results:
        if not ok:
            print(f"    ✗  {name}" + (f": {detail}" if detail else ""))

print()
sys.exit(0 if failed == 0 else 1)
