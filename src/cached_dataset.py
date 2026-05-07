"""
Cached Feature Dataset
----------------------
Loads pre-computed features from disk for fast MLP-only training.
No CNN forward pass needed — each __getitem__ is a simple .pt load.

Scene descriptor is recomputed from cached features (not saved).
"""

import os
import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Dataset that loads pre-computed .pt feature files."""

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
        self._files = sorted(
            f for f in os.listdir(cache_dir) if f.endswith(".pt")
        )
        if len(self._files) == 0:
            raise FileNotFoundError(f"No .pt files in {cache_dir}")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> dict:
        path = os.path.join(self._cache_dir, self._files[index])
        data = torch.load(path, weights_only=True)

        return {
            "features": data["features"],       # [N, 128]
            "class_ids": data["class_ids"],      # [N]
            "labels": data["labels"],            # [N]
            "task_id": data["task_id"],           # int
        }


def cached_collate_fn(batch: list) -> list:
    """Custom collate — variable N per sample."""
    return batch
