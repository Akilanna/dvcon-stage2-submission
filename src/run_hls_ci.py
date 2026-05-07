"""
Automated CI build script for DVCON HLS accelerator.

Flow:
1) Export trained PyTorch weights to HLS header.
2) Run Vitis HLS using run_hls.tcl.
3) Parse csynth report.
4) Print latency and resource summary.

Usage:
  py scripts/run_hls_ci.py
"""

import re
import shutil
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Optional


def run_cmd(cmd: list[str], cwd: Optional[Path] = None) -> None:
    """Run command and raise on failure."""
    pretty_cwd = str(cwd) if cwd else str(Path.cwd())
    print(f"\n[RUN] cwd={pretty_cwd}")
    print("[RUN] " + " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={result.returncode}): {' '.join(cmd)}"
        )


def resolve_hls_launcher() -> tuple[str, Path]:
    """Resolve HLS launcher.

    Returns:
        (kind, path)
        kind in {"vitis_hls", "vitis_run", "vitis"}
    """
    # 1) Explicit env override.
    env_path = os.environ.get("VITIS_HLS")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return "vitis_hls", p

    # 2) PATH lookup.
    which = shutil.which("vitis_hls")
    if which:
        return "vitis_hls", Path(which)

    # 3) Known install locations, including user-provided Vitis 2024.2 path.
    candidates = [
        Path(r"D:\Xilinx\Vitis\2024.2\bin\vitis_hls.bat"),
        Path(r"D:\Xilinx\Vitis\2024.2\bin\vitis_hls.exe"),
        Path(r"C:\Xilinx\Vitis\2024.2\bin\vitis_hls.bat"),
        Path(r"C:\Xilinx\Vitis\2024.2\bin\vitis_hls.exe"),
    ]
    for p in candidates:
        if p.is_file():
            return "vitis_hls", p

    # 4) If vitis.vbs exists, try neighboring vitis_hls launchers.
    vbs_candidates = [
        Path(r"D:\Xilinx\Vitis\2024.2\bin\vitis.vbs"),
        Path(r"C:\Xilinx\Vitis\2024.2\bin\vitis.vbs"),
    ]
    for vbs in vbs_candidates:
        if vbs.is_file():
            for name in ("vitis_hls.bat", "vitis_hls.exe"):
                p = vbs.parent / name
                if p.is_file():
                    return "vitis_hls", p

    # 5) Prefer vitis-run CLI (2024.2+) for HLS Tcl flow.
    vitis_run_candidates = [
        Path(r"D:\Xilinx\Vitis\2024.2\bin\vitis-run.bat"),
        Path(r"D:\Xilinx\Vitis\2024.2\bin\vitis-run"),
        Path(r"C:\Xilinx\Vitis\2024.2\bin\vitis-run.bat"),
        Path(r"C:\Xilinx\Vitis\2024.2\bin\vitis-run"),
    ]
    for p in vitis_run_candidates:
        if p.is_file():
            return "vitis_run", p

    # 6) Fallback to Vitis unified launcher.
    vitis_candidates = [
        Path(r"D:\Xilinx\Vitis\2024.2\bin\vitis.bat"),
        Path(r"D:\Xilinx\Vitis\2024.2\bin\vitis"),
        Path(r"C:\Xilinx\Vitis\2024.2\bin\vitis.bat"),
        Path(r"C:\Xilinx\Vitis\2024.2\bin\vitis"),
    ]
    for p in vitis_candidates:
        if p.is_file():
            return "vitis", p

    raise FileNotFoundError(
        "Could not find vitis_hls launcher. Set VITIS_HLS env var to full path "
        "(e.g., D:\\Xilinx\\Vitis\\2024.2\\bin\\vitis_hls.bat)."
    )


def build_hls_cmd(vitis_hls_path: Path, tcl_path: Path) -> list[str]:
    """Build command to invoke Vitis HLS robustly on Windows for .bat or .exe."""
    suffix = vitis_hls_path.suffix.lower()
    if suffix == ".bat":
        return ["cmd", "/c", str(vitis_hls_path), "-f", str(tcl_path)]
    return [str(vitis_hls_path), "-f", str(tcl_path)]


def build_vitis_hls_cmd(vitis_path: Path, tcl_path: Path) -> list[str]:
    """Build command to invoke HLS mode via Vitis unified launcher."""
    suffix = vitis_path.suffix.lower()
    args = [str(vitis_path), "-mode", "hls", "-source", str(tcl_path)]
    if suffix == ".bat":
        return ["cmd", "/c", *args]
    return args


def build_vitis_run_hls_cmd(vitis_run_path: Path, tcl_path: Path) -> list[str]:
    """Build command to invoke HLS Tcl flow via vitis-run."""
    suffix = vitis_run_path.suffix.lower()
    args = [str(vitis_run_path), "--mode", "hls", "--tcl", str(tcl_path)]
    if suffix == ".bat":
        return ["cmd", "/c", *args]
    return args


def parse_latency_and_ii(report_text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse latency (cycles) and II from csynth report.

    Tries table form first, then falls back to regex.
    """
    lines = report_text.splitlines()

    latency = None
    ii = None

    # Try to parse from "Latency (clock cycles)" summary table.
    for idx, line in enumerate(lines):
        if "Latency (clock cycles)" in line:
            window = lines[idx : min(idx + 40, len(lines))]
            for w in window:
                # Typical row: | 25060| 25060| 25061| 25061|  none |
                m = re.match(
                    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|",
                    w.strip(),
                )
                if m:
                    latency = m.group(1)  # min latency
                    ii = m.group(3)       # min interval
                    return latency, ii

    # Fallback: "Final II: X" from scheduler diagnostics.
    m_ii = re.search(r"Final\s+II\s*:\s*(\d+)", report_text)
    if m_ii:
        ii = m_ii.group(1)

    # Fallback: "Latency" with numbers on same line.
    m_lat = re.search(r"Latency[^\n]*?(\d+)", report_text)
    if m_lat:
        latency = m_lat.group(1)

    return latency, ii


def parse_resource_usage(report_text: str) -> Dict[str, str]:
    """
    Parse Total resource row from csynth utilization table.

    Supports headers with either DSP or DSP48E naming.
    """
    lines = report_text.splitlines()

    header_idx = None
    headers: list[str] = []

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("|") and "BRAM_18K" in s and "LUT" in s:
            cols = [c.strip() for c in s.strip("|").split("|")]
            if "Name" in cols:
                header_idx = i
                headers = cols
                break

    if header_idx is None:
        return {"BRAM_18K": "N/A", "DSP48E": "N/A", "LUT": "N/A"}

    # Find Total row below header.
    total_row: Optional[list[str]] = None
    for j in range(header_idx + 1, min(header_idx + 50, len(lines))):
        s = lines[j].strip()
        if not s.startswith("|"):
            continue
        cols = [c.strip() for c in s.strip("|").split("|")]
        if cols and cols[0] == "Total":
            total_row = cols
            break

    if total_row is None:
        return {"BRAM_18K": "N/A", "DSP48E": "N/A", "LUT": "N/A"}

    index = {name: k for k, name in enumerate(headers)}

    def get_col(name: str) -> str:
        k = index.get(name)
        if k is None or k >= len(total_row):
            return "N/A"
        return total_row[k]

    dsp_val = get_col("DSP48E")
    if dsp_val == "N/A":
        dsp_val = get_col("DSP")

    return {
        "BRAM_18K": get_col("BRAM_18K"),
        "DSP48E": dsp_val,
        "LUT": get_col("LUT"),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    exporter = repo_root / "scripts" / "export_hls_weights.py"
    ckpt = repo_root / "checkpoints" / "ranker_best.pt"
    hls_dir = repo_root / "fpga" / "hls" / "ranking_mlp"
    hls_tcl = hls_dir / "run_hls.tcl"
    rpt = hls_dir / "ranking_mlp_hls" / "solution1" / "syn" / "report" / "ranking_mlp_top_csynth.rpt"

    if not exporter.is_file():
        raise FileNotFoundError(f"Exporter script not found: {exporter}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not hls_tcl.is_file():
        raise FileNotFoundError(f"HLS Tcl not found: {hls_tcl}")

    print("=== DVCON HLS CI ===")

    # Step 1: Export weights.
    run_cmd(
        [
            sys.executable,
            str(exporter),
            "--ckpt",
            str(ckpt),
        ],
        cwd=repo_root,
    )

    # Step 2: Run HLS synthesis.
    launcher_kind, launcher_path = resolve_hls_launcher()
    if launcher_kind == "vitis_hls":
        hls_cmd = build_hls_cmd(launcher_path, hls_tcl)
    elif launcher_kind == "vitis_run":
        hls_cmd = build_vitis_run_hls_cmd(launcher_path, hls_tcl)
    else:
        hls_cmd = build_vitis_hls_cmd(launcher_path, hls_tcl)
    run_cmd(
        hls_cmd,
        cwd=hls_dir,
    )

    # Step 3: Parse synthesis report.
    if not rpt.is_file():
        raise FileNotFoundError(f"Synthesis report not found: {rpt}")

    report_text = rpt.read_text(encoding="utf-8", errors="ignore")
    latency, ii = parse_latency_and_ii(report_text)
    res = parse_resource_usage(report_text)

    # Step 4: Print summary.
    print("\nHLS SYNTHESIS SUMMARY")
    print(f"Latency (cycles): {latency if latency is not None else 'N/A'}")
    print(f"Initiation Interval: {ii if ii is not None else 'N/A'}")
    print(f"DSP48E: {res['DSP48E']}")
    print(f"BRAM_18K: {res['BRAM_18K']}")
    print(f"LUT: {res['LUT']}")
    print("---------")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
