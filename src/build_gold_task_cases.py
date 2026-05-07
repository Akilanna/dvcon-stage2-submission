#!/usr/bin/env python3
"""Build a gold positive test case CSV for the 14 paper tasks.

This version mines the official COCO-Tasks annotation files directly. It uses
preferred annotations (category_id == 1) as the source of truth and writes cases
that are already task-aligned, so the resulting validation set is much cleaner
than heuristic detector-based mining.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

TASK_RELEVANT_CLASSES = {
    0: ["chair", "couch", "bed", "bench", "dining table"],
    1: ["chair", "couch", "bed", "toilet", "bench", "dining table"],
    2: ["vase", "potted plant", "bottle", "cup", "bowl"],
    3: ["fork", "spoon", "knife"],
    4: ["bottle", "cup", "bowl", "sink"],
    5: ["spoon", "fork", "knife", "cup", "bowl"],
    6: ["spoon", "fork"],
    7: ["bottle", "knife", "spoon"],
    8: ["scissors", "knife"],
    9: ["wine glass", "bottle", "cup"],
    10: ["bottle", "cup", "bowl", "spoon"],
    11: ["knife", "spoon"],
    12: ["fire hydrant", "bottle", "cup", "bowl", "sink"],
    13: ["baseball bat", "tennis racket"],
}

COCO_CATEGORY_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    85: "book",
    86: "clock",
    87: "vase",
    88: "scissors",
    89: "teddy bear",
    90: "hair drier",
    91: "toothbrush",
}


def load_annotation_files(annotation_dir: str):
    """Return train/test task files if available."""
    candidates = []
    for split in ("train", "test"):
        for task_id in range(1, 15):
            path = os.path.join(annotation_dir, f"task_{task_id}_{split}.json")
            if os.path.isfile(path):
                candidates.append((task_id - 1, split, path))
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Build known-positive task cases CSV")
    parser.add_argument("--annotation-dir", default="data/annotations", help="Directory with task_*.json files")
    parser.add_argument("--output-csv", default="targeted_task_cases_gold.csv", help="Output cases CSV")
    parser.add_argument("--summary", default="targeted_task_cases_gold_summary.txt", help="Output summary text")
    parser.add_argument("--per-task", type=int, default=5, help="Target positives per task")
    args = parser.parse_args()

    files = load_annotation_files(args.annotation_dir)
    if not files:
        raise FileNotFoundError(f"No task annotation files found in {args.annotation_dir}")

    # Map COCO category ids to names.
    from inference.run_pipeline import COCO_CLASSES

    selected = defaultdict(list)
    used_images = {tid: set() for tid in range(14)}
    stats = defaultdict(int)

    for task_id, split, path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_lookup = {img["id"]: img["file_name"] for img in data.get("images", [])}
        grouped = defaultdict(list)
        for ann in data.get("annotations", []):
            grouped[ann["image_id"]].append(ann)

        for image_id, anns in grouped.items():
            if len(selected[task_id]) >= args.per_task:
                break

            preferred = [ann for ann in anns if ann.get("category_id") == 1]
            if not preferred:
                continue

            file_name = image_lookup.get(image_id)
            if not file_name:
                continue

            if file_name in used_images[task_id]:
                continue

            image_path = os.path.join("data/coco/train2014" if "train" in split else "data/coco/val2014", file_name)
            expected_classes = sorted(
                {
                    COCO_CATEGORY_ID_TO_NAME.get(int(ann["COCO_category_id"]), f"coco_{int(ann['COCO_category_id'])}")
                    for ann in preferred
                }
            )

            selected[task_id].append((image_path, expected_classes, split))
            used_images[task_id].add(file_name)
            stats[f"{task_id}:{split}"] += 1

    rows = []
    for task_id in range(14):
        for image_path, expected_classes, split in selected[task_id]:
            rows.append(
                {
                    "image_path": image_path.replace("\\", "/"),
                    "task_id": task_id,
                    "expected_classes": "|".join(expected_classes),
                    "split": split,
                }
            )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "task_id", "expected_classes", "split"])
        writer.writeheader()
        writer.writerows(rows)

    timestamp = datetime.now().isoformat()
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("GOLD CASES BUILD SUMMARY\n")
        f.write("=" * 36 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Annotation dir: {args.annotation_dir}\n")
        f.write(f"Per-task target: {args.per_task}\n")
        f.write(f"Total cases written: {len(rows)}\n")
        f.write(f"Output CSV: {args.output_csv}\n\n")
        for task_id in range(14):
            f.write(f"Task {task_id:02d}: {len(selected[task_id])} cases\n")

    print("\n" + "=" * 50)
    print(f"Cases written: {len(rows)}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Summary: {args.summary}")
    print("=" * 50)


if __name__ == "__main__":
    main()
