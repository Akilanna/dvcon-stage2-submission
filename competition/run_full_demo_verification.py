#!/usr/bin/env python3
"""Run full verification over the curated demo manifest.

Outputs:
- competition/full_demo_verification.csv
- competition/full_demo_summary.json
- competition/demo_outputs/*.png
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from competition.competition_settings import get_allowed, get_threshold
from verify_model import ModelVerifier, COCO_CLASSES


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "competition" / "demo_manifest.json"
CSV_OUT = ROOT / "competition" / "full_demo_verification.csv"
SUMMARY_OUT = ROOT / "competition" / "full_demo_summary.json"
OUTPUT_DIR = ROOT / "competition" / "demo_outputs"
CHECKPOINT = ROOT / "checkpoints" / "competition_14task_final" / "ranker_best.pt"


def _load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = _load_manifest()
    verifier = ModelVerifier(ranker_ckpt=str(CHECKPOINT), device="cpu")

    rows = []
    failures = []
    per_task = defaultdict(int)

    for idx, entry in enumerate(entries):
        image_path = entry["image_path"]
        task_id = int(entry["task_id"])
        out_png = OUTPUT_DIR / f"task{task_id:02d}_{idx:03d}.png"
        runtime_ok = True
        try:
            result = verifier.verify(
                image_path,
                task_id,
                score_threshold=float(get_threshold(task_id)),
                apply_compatibility_filter=True,
            )
            verifier.visualize(result, str(out_png))

            best_idx = result.get("best_index")
            prediction = None
            allowed_prediction = ""
            if best_idx is not None:
                class_id = int(result["class_id"])
                prediction = COCO_CLASSES[class_id]
                if not result.get("no_appropriate_object", False) and result.get("is_compatible", False):
                    allowed_prediction = prediction

            rows.append({
                "task_id": task_id,
                "image_path": image_path,
                "prediction": prediction or result.get("reject_message") or "No valid affordance candidate detected",
                "confidence": float(result.get("best_score", 0.0)),
                "rejected": bool(result.get("no_appropriate_object", False)),
                "allowed_prediction": allowed_prediction,
                "runtime_ok": runtime_ok,
                "output_png": str(out_png),
            })
            per_task[task_id] += 1

        except Exception as exc:
            runtime_ok = False
            failures.append({
                "task_id": task_id,
                "image_path": image_path,
                "error": str(exc),
            })
            rows.append({
                "task_id": task_id,
                "image_path": image_path,
                "prediction": "No valid affordance candidate detected",
                "confidence": 0.0,
                "rejected": True,
                "allowed_prediction": "",
                "runtime_ok": runtime_ok,
                "output_png": str(out_png),
            })

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "manifest": str(MANIFEST),
        "checkpoint": str(CHECKPOINT),
        "total_entries": len(rows),
        "failures": failures,
        "task_counts": {str(k): int(v) for k, v in per_task.items()},
        "rejected_count": sum(1 for r in rows if r["rejected"]),
        "runtime_ok_count": sum(1 for r in rows if r["runtime_ok"]),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()