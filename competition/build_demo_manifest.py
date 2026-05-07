#!/usr/bin/env python3
"""Build a curated 14-task demo manifest from the cleaned gold2 task cases.

The goal is to keep 3-5 clear, semantically reasonable images per task for
competition demos. This script selects up to four unique images per task from
targeted_task_cases_gold2.csv and writes competition/demo_manifest.json.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "targeted_task_cases_gold.csv"
OUTPUT = ROOT / "competition" / "demo_manifest.json"


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def main() -> None:
    rows = load_rows(SOURCE)
    grouped = defaultdict(list)

    for row in rows:
        task_id = int(row["task_id"])
        image_path = row["image_path"]
        expected_classes = [part for part in row["expected_classes"].split("|") if part]
        grouped[task_id].append(
            {
                "task_id": task_id,
                "image_path": image_path,
                "expected_classes": expected_classes,
                "source": "targeted_task_cases_gold2.csv",
            }
        )

    manifest = []
    per_task_counts = {}
    for task_id in range(14):
        seen = set()
        selected = []
        for item in grouped.get(task_id, []):
            if item["image_path"] in seen:
                continue
            seen.add(item["image_path"])
            selected.append(item)
            if len(selected) >= 4:
                break
        per_task_counts[task_id] = len(selected)
        manifest.extend(selected)

    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT}")
    for task_id in range(14):
        print(f"task {task_id}: {per_task_counts.get(task_id, 0)} demo images")


if __name__ == "__main__":
    main()