#!/usr/bin/env python3
"""Build cached features for the balanced competition manifest.

For each manifest entry, run the YOLO detector, extract object features,
and write a cached .pt file with detections, labels, and task_id preserved.
Failures are logged to data/cache_build_summary.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.detector.yolo_wrapper import YOLODetector
from models.feature_module.feature_extractor import FeatureExtractor


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "competition_multitask_manifest.json"
OUTPUT_DIR = ROOT / "data" / "cached_features_competition_14task"
SUMMARY_PATH = ROOT / "data" / "cache_build_summary.json"


def _safe_name(image_path: str, task_id: int, copy_index: int) -> str:
    stem = Path(image_path).stem
    return f"task{task_id:02d}_{copy_index:03d}_{stem}.pt"


def _load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def main() -> None:
    manifest = _load_manifest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(model_name="yolov8n.pt", device="cpu", conf_threshold=0.20)
    extractor = FeatureExtractor(feature_dim=128, device="cpu")
    extractor.eval()

    failures: list[dict] = []
    cached = 0
    skipped = 0

    for idx, entry in enumerate(manifest):
        image_path = ROOT / entry["image_path"]
        task_id = int(entry["task_id"])
        expected_objects = list(entry.get("expected_objects", []))
        copy_index = int(entry.get("copy_index", idx))
        out_path = OUTPUT_DIR / _safe_name(entry["image_path"], task_id, copy_index)

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")

            det = detector.detect(image)
            boxes = det["boxes"]
            class_ids = det["class_ids"]
            scores = det.get("scores")

            if boxes is None or len(boxes) == 0:
                raise RuntimeError("No detections")

            with torch.no_grad():
                features = extractor(image, boxes)

            class_names = []
            labels = []
            for cid in class_ids.tolist():
                from verify_model import COCO_CLASSES
                cls_name = COCO_CLASSES[int(cid)]
                class_names.append(cls_name)
                labels.append(1.0 if cls_name in expected_objects else 0.0)

            if sum(labels) == 0:
                raise RuntimeError(f"No expected object detected; expected={expected_objects}")

            tmp_path = str(out_path) + ".tmp"
            torch.save(
                {
                    "features": features.cpu(),
                    "boxes": boxes.cpu(),
                    "class_ids": class_ids.cpu(),
                    "class_names": class_names,
                    "labels": torch.tensor(labels, dtype=torch.float32),
                    "task_id": task_id,
                    "image_path": entry["image_path"],
                    "image_shape": image.shape[:2],
                    "expected_objects": expected_objects,
                    "detection_scores": scores.cpu() if scores is not None else None,
                },
                tmp_path,
            )
            os.replace(tmp_path, out_path)
            cached += 1

        except Exception as exc:
            skipped += 1
            failures.append({
                "index": idx,
                "task_id": task_id,
                "image_path": entry["image_path"],
                "expected_objects": expected_objects,
                "error": str(exc),
            })
            print(f"[FAIL] task {task_id} {entry['image_path']}: {exc}")

        if (idx + 1) % 20 == 0:
            print(f"[{idx + 1}/{len(manifest)}] cached={cached} skipped={skipped}")

    summary = {
        "manifest": str(MANIFEST),
        "output_dir": str(OUTPUT_DIR),
        "total_entries": len(manifest),
        "cached_entries": cached,
        "skipped_entries": skipped,
        "failures": failures,
        "task_counts": {},
    }

    for entry in manifest:
        tid = str(int(entry["task_id"]))
        summary["task_counts"][tid] = summary["task_counts"].get(tid, 0) + 1

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote cache summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()