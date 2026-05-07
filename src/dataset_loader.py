"""
COCO-Tasks Dataset Loader
--------------------------
Parses COCO-Tasks annotations and serves training samples for the RankingMLP.

Real COCO-Tasks annotation format (per-task files: task_{1-14}_{train,test}.json):
{
  "images": [ {"id": int, "file_name": str, ...} ],
  "annotations": [
    {
      "image_id": int,
      "bbox": [x, y, w, h],      # COCO format
      "COCO_category_id": int,    # actual COCO class (e.g. 58)
      "category_id": int,         # 0 = not preferred, 1 = preferred (the label)
      "id": int
    }
  ]
}

Task ID is implicit from filename (task_1 → task_id=0, ..., task_14 → task_id=13).
Train images: COCO_train2014_*.jpg  |  Test images: COCO_val2014_*.jpg

Each __getitem__ returns one (image, task) pair with all its object annotations.
"""

import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class COCOTasksDataset(Dataset):
    """PyTorch dataset for COCO-Tasks training samples."""

    def __init__(
        self,
        annotation_dir: str,
        image_root: str,
        task_ids: list = None,
        split: str = "train",
    ):
        """
        Args:
            annotation_dir: Directory containing task_{1-14}_{train,test}.json files.
            image_root: Directory containing COCO images (train2014/ or val2014/).
            task_ids: List of task IDs to load (0-based, 0–13). None = all 14 tasks.
            split: "train" or "test".
        """
        if task_ids is None:
            task_ids = list(range(14))

        self._image_root = image_root
        self._image_lookup = {}
        self._samples = []

        for task_id in task_ids:
            # task_id 0 → task_1_train.json, task_id 13 → task_14_train.json
            fname = f"task_{task_id + 1}_{split}.json"
            ann_path = os.path.join(annotation_dir, fname)
            if not os.path.isfile(ann_path):
                raise FileNotFoundError(f"Annotation file not found: {ann_path}")

            with open(ann_path, "r") as f:
                data = json.load(f)

            # Merge image lookup from this file
            for img in data["images"]:
                self._image_lookup[img["id"]] = img["file_name"]

            # Group annotations by image_id
            grouped = defaultdict(list)
            for ann in data["annotations"]:
                grouped[ann["image_id"]].append(ann)

            for img_id, anns in grouped.items():
                if len(anns) > 0:
                    self._samples.append((img_id, task_id, anns))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict:
        img_id, task_id, anns = self._samples[index]

        # Load image
        fname = self._image_lookup[img_id]
        img_path = os.path.join(self._image_root, fname)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # Parse annotations — real COCO-Tasks format
        boxes = []
        class_ids = []
        labels = []

        for ann in anns:
            # COCO bbox is [x, y, w, h] → convert to [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            class_ids.append(ann["COCO_category_id"])   # actual COCO class
            labels.append(ann["category_id"])            # 0=not preferred, 1=preferred

        return {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "class_ids": torch.tensor(class_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "task_id": task_id,
        }


def collate_fn(batch: list) -> list:
    """
    Custom collate that returns a list of sample dicts.

    Each image may have a different number of objects (N varies),
    so standard stacking is not applicable.
    """
    return batch


if __name__ == "__main__":
    import sys

    ann_dir = "data/annotations"
    img_root = "data/coco/train2014"  # COCO 2014 train images

    # Quick check: load task 1 only
    ds = COCOTasksDataset(ann_dir, img_root, task_ids=[0], split="train")
    print(f"Dataset size (task 1 train): {len(ds)}")

    # Don't try to load images if they don't exist yet — just verify parsing
    sample_meta = ds._samples[0]
    img_id, task_id, anns = sample_meta
    print(f"Sample: image_id={img_id}, task_id={task_id}, num_objects={len(anns)}")
    print(f"First ann: COCO_class={anns[0]['COCO_category_id']}, "
          f"label={anns[0]['category_id']}, bbox={anns[0]['bbox']}")

    # All-tasks test
    ds_all = COCOTasksDataset(ann_dir, img_root, task_ids=None, split="train")
    print(f"\nAll 14 tasks combined: {len(ds_all)} samples")
