#!/usr/bin/env python3
"""Build a balanced 14-task competition manifest.

Inputs:
- targeted_task_cases_gold.csv
- targeted_task_cases_gold2.csv
- competition/demo_manifest.json

Output:
- data/competition_multitask_manifest.json

The manifest keeps only semantically believable classes for each task, then
pads each task to the target sample count by cycling through the curated set.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from competition.competition_settings import get_allowed


ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = [
    ROOT / "targeted_task_cases_gold.csv",
    ROOT / "targeted_task_cases_gold2.csv",
    ROOT / "competition" / "demo_manifest.json",
]
OUTPUT = ROOT / "data" / "competition_multitask_manifest.json"

TARGET_PER_TASK = 10


def _normalize_expected_objects(task_id: int, expected_objects: list[str]) -> list[str]:
    allowed = set(get_allowed(task_id))
    return [obj for obj in expected_objects if obj in allowed]


def _load_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = []
        for item in raw:
            task_id = int(item["task_id"])
            expected = item.get("expected_objects") or item.get("expected_classes") or []
            if isinstance(expected, str):
                expected = [part for part in expected.split("|") if part]
            expected = _normalize_expected_objects(task_id, list(expected))
            if not expected:
                continue
            entries.append({
                "task_id": task_id,
                "image_path": item["image_path"],
                "expected_objects": expected,
                "source": path.name,
            })
        return entries

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        entries = []
        for row in reader:
            task_id = int(row["task_id"])
            expected = [part for part in row["expected_classes"].split("|") if part]
            expected = _normalize_expected_objects(task_id, expected)
            if not expected:
                continue
            entries.append({
                "task_id": task_id,
                "image_path": row["image_path"],
                "expected_objects": expected,
                "source": path.name,
            })
        return entries


def main() -> None:
    grouped: dict[int, list[dict]] = defaultdict(list)
    seen = set()

    for source in SOURCE_FILES:
        for entry in _load_entries(source):
            key = (entry["task_id"], entry["image_path"], tuple(entry["expected_objects"]))
            if key in seen:
                continue
            seen.add(key)
            grouped[entry["task_id"]].append(entry)

    manifest: list[dict] = []
    per_task_counts: dict[int, int] = {}

    for task_id in range(14):
        candidates = grouped.get(task_id, [])
        if not candidates:
            raise SystemExit(f"No curated candidates found for task {task_id}")

        selected = []
        while len(selected) < TARGET_PER_TASK:
            for idx, item in enumerate(candidates):
                if len(selected) >= TARGET_PER_TASK:
                    break
                copy_index = len(selected)
                cloned = dict(item)
                cloned["copy_index"] = copy_index
                selected.append(cloned)
            if len(selected) < TARGET_PER_TASK:
                # Cycle again to pad the task. Repetition is intentional here:
                # the demo/training set must be balanced even when only a small
                # curated pool is available for a task.
                continue

        per_task_counts[task_id] = len(selected)
        manifest.extend(selected)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT}")
    for task_id in range(14):
        print(f"task {task_id}: {per_task_counts[task_id]} entries")


if __name__ == "__main__":
    main()