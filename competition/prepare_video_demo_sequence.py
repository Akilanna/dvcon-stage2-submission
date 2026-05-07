#!/usr/bin/env python3
"""Prepare a simple ordered demo sequence for video / slides."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "competition" / "full_demo_verification.csv"
OUT_PATH = ROOT / "competition" / "video_demo_sequence.md"


def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    successful = [r for r in rows if r["runtime_ok"].lower() == "true" and r["rejected"].lower() == "false"]
    rejected = [r for r in rows if r["rejected"].lower() == "true"]
    masking_example = "reports/slides/slide07_false_accept.png"

    lines = []
    lines.append("# Competition Demo Sequence")
    lines.append("")
    lines.append("1. Intro/title")
    lines.append("2. Architecture overview")
    lines.append("3. Task-conditioned examples")
    for row in successful[:14]:
        lines.append(
            f"   - Task {row['task_id']}: {Path(row['image_path']).name} -> {row['prediction']} "
            f"(confidence {float(row['confidence']):.3f})"
        )
    if rejected:
        row = rejected[0]
        lines.append(
            f"4. Reject/no-valid example: task {row['task_id']} on {Path(row['image_path']).name}"
        )
    else:
        lines.append("4. Reject/no-valid example: use a curated abstain case from verification")
    lines.append(f"5. Masking false-accept limitation: {masking_example}")
    lines.append("6. Conclusion")
    lines.append("")
    lines.append("Primary flow uses only curated successful examples.")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()