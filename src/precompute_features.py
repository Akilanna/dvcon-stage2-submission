"""
Pre-compute features for all COCO-Tasks samples.

Runs the frozen ResNet18 feature extractor + scene descriptor ONCE
on every (image, task) sample and saves the results as .pt files.

This eliminates redundant CNN computation during training — each
epoch of MLP training becomes ~100x faster.

Output per sample (saved as dict in .pt file):
    - features:   [N, 128]   object visual features
    - scene:      [128]       scene descriptor (max-pool)
    - class_ids:  [N]         COCO category IDs
    - labels:     [N]         binary relevance (0/1)
    - task_id:    int         task index (0-13)

Usage: py scripts/precompute_features.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models.feature_module.feature_extractor import FeatureExtractor
from models.feature_module.scene_descriptor import SceneDescriptor
from training.dataset_loader import COCOTasksDataset


def precompute(
    annotation_dir: str = "data/annotations",
    image_root: str = "data/coco/train2014",
    output_dir: str = "data/cached_features",
    feature_dim: int = 128,
    device: str = "cpu",
):
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = COCOTasksDataset(annotation_dir, image_root)
    n = len(dataset)
    print(f"Dataset: {n} samples")

    # Load frozen feature extractor
    feature_extractor = FeatureExtractor(feature_dim=feature_dim, device=device)
    feature_extractor.eval()
    scene_descriptor = SceneDescriptor()

    # Check for existing progress
    existing = set(f for f in os.listdir(output_dir) if f.endswith(".pt"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"Resuming from sample {start_idx} ({start_idx} already cached)")

    t0 = time.time()
    skipped = 0

    for i in range(start_idx, n):
        out_path = os.path.join(output_dir, f"sample_{i:06d}.pt")

        sample = dataset[i]
        boxes = sample["boxes"]

        if boxes.shape[0] == 0:
            skipped += 1
            continue

        with torch.no_grad():
            features = feature_extractor(sample["image"], boxes)
            scene = scene_descriptor.compute(features)

        torch.save({
            "features": features.cpu(),
            "scene": scene.cpu(),
            "class_ids": sample["class_ids"],
            "labels": sample["labels"],
            "task_id": sample["task_id"],
        }, out_path)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - start_idx) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {rate:.1f} samples/s | "
                  f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s | "
                  f"skipped={skipped}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone. {n - skipped} samples cached in {elapsed:.0f}s")
    print(f"Output: {output_dir}/")
    print(f"Skipped (empty boxes): {skipped}")


if __name__ == "__main__":
    precompute()
