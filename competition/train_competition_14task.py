#!/usr/bin/env python3
"""Balanced 14-task competition fine-tune from cached features.

This script starts from the current best checkpoint, trains for a small number
of CPU-safe epochs, and saves logs plus the best competition checkpoint.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset

from competition.competition_settings import COMPATIBILITY_PENALTY, get_allowed, get_threshold
from models.ranker.ranking_mlp import RankingMLP, build_ranking_input
from models.task_encoder.task_embedding import TaskEmbeddingManager
from models.feature_module.scene_descriptor import SceneDescriptor
from training.cached_dataset import CachedFeatureDataset, cached_collate_fn
from training.loss_functions import ranking_loss_softmax, ranking_loss_topk_hardneg_gated


ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cached_features_competition_14task"
CHECKPOINT_IN = ROOT / "checkpoints" / "task_14_final" / "ranker_best.pt"
CHECKPOINT_OUT = ROOT / "checkpoints" / "competition_14task_final"
LOG_DIR = ROOT / "logs" / "competition_14task_final"
EMBEDDING_FILE = ROOT / "data" / "embeddings" / "tasks.npy"

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


def _load_ckpt(model: RankingMLP, scene: SceneDescriptor, optimizer, path: Path, device: str):
    if not path.exists():
        raise FileNotFoundError(f"Starting checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "scene_descriptor" in ckpt:
            scene.load_state_dict(ckpt["scene_descriptor"])
        if optimizer is not None and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise RuntimeError("Unsupported checkpoint format")


def _task_split_indices(dataset: CachedFeatureDataset, seed: int = 42, val_ratio: float = 0.15):
    task_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        task_to_indices[int(sample["task_id"])].append(idx)

    train_indices = []
    val_indices = []
    rng = random.Random(seed)
    for task_id in range(14):
        indices = task_to_indices.get(task_id, [])
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio)) if indices else 0
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    return train_indices, val_indices, task_to_indices


def _interleave_by_task(dataset: CachedFeatureDataset, indices: list[int]) -> list[int]:
    buckets = defaultdict(list)
    for idx in indices:
        buckets[int(dataset[idx]["task_id"])].append(idx)

    max_len = max((len(v) for v in buckets.values()), default=0)
    ordered = []
    for round_idx in range(max_len):
        for task_id in range(14):
            bucket = buckets.get(task_id, [])
            if not bucket:
                continue
            ordered.append(bucket[round_idx % len(bucket)])
    return ordered


def _run_epoch(loader, ranker, scene, task_mgr, optimizer=None, epoch=None, compat_scale=0.08):
    is_train = optimizer is not None
    ranker.train(is_train)
    scene.train(is_train)
    total_loss = 0.0
    total_items = 0
    task_counts = defaultdict(int)

    for batch in loader:
        sample = batch[0]
        features = sample["features"]
        class_ids = sample["class_ids"]
        labels = sample["labels"]
        task_id = int(sample["task_id"])

        if features.shape[0] == 0:
            continue

        task_counts[task_id] += 1
        task_embed = task_mgr.get_embedding(task_id).to(features.device)
        scene_feat = scene.compute(features, task_embed)
        obj_feat, scn_feat, task_emb, aff_feat = build_ranking_input(
            features,
            scene_feat,
            task_embed,
            class_ids,
            boxes=sample.get("boxes"),
            image_shape=sample.get("image_shape"),
            task_id=task_id,
        )

        logits = ranker(
            obj_feat,
            scn_feat,
            task_emb,
            aff_feat,
            class_ids=class_ids,
            compat_scale=compat_scale,
            task_id=task_id,
        )

        pos_mask = labels > 0.5
        if pos_mask.any():
            pos_idx = int(pos_mask.nonzero(as_tuple=True)[0][0].item())
            target = torch.tensor([pos_idx], device=logits.device, dtype=torch.long)
            ce_loss = F.cross_entropy(logits.unsqueeze(0), target)
            hardneg_loss = ranking_loss_topk_hardneg_gated(logits.squeeze(), pos_idx, margin=0.5, k=3)

            allowed = get_allowed(task_id)
            incompat_penalty = torch.tensor(0.0, device=logits.device)
            if allowed:
                incompatible_mask = torch.tensor(
                    [COCO_CLASSES[int(cid)] not in allowed for cid in class_ids.tolist()],
                    device=logits.device,
                    dtype=torch.bool,
                )
                if incompatible_mask.any():
                    incompat_penalty = torch.relu(logits[incompatible_mask] + 0.20).mean()

            loss = 0.45 * ce_loss + 1.0 * hardneg_loss + 0.35 * incompat_penalty
        else:
            loss = ranking_loss_softmax(logits, labels.to(logits.device))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_items += 1

    return {
        "loss": total_loss / max(total_items, 1),
        "n": total_items,
        "task_counts": {str(k): int(v) for k, v in task_counts.items()},
    }


def main(epochs: int = 5, lr: float = 1e-4, val_ratio: float = 0.15):
    CHECKPOINT_OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(min(6, os.cpu_count() or 6))
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"Cache directory not found: {CACHE_DIR}")

    dataset = CachedFeatureDataset(str(CACHE_DIR))
    if len(dataset) == 0:
        raise SystemExit("No cached features available for competition training")

    train_indices, val_indices, task_buckets = _task_split_indices(dataset, val_ratio=val_ratio)
    train_indices = _interleave_by_task(dataset, train_indices)
    val_indices = _interleave_by_task(dataset, val_indices)

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=cached_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=cached_collate_fn)

    scene = SceneDescriptor(use_gating=True)
    task_mgr = TaskEmbeddingManager(str(EMBEDDING_FILE))
    ranker = RankingMLP()

    optimizer = torch.optim.Adam(list(ranker.parameters()) + list(scene.parameters()), lr=lr * 0.3)
    _load_ckpt(ranker, scene, optimizer, CHECKPOINT_IN, device="cpu")

    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = _run_epoch(train_loader, ranker, scene, task_mgr, optimizer=optimizer, epoch=epoch)
        with torch.no_grad():
            val_stats = _run_epoch(val_loader, ranker, scene, task_mgr, optimizer=None, epoch=epoch)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_n": train_stats["n"],
            "val_n": val_stats["n"],
            "train_task_counts": train_stats["task_counts"],
            "val_task_counts": val_stats["task_counts"],
            "epoch_seconds": elapsed,
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_stats['loss']:.4f} | val_loss={val_stats['loss']:.4f} | "
            f"train_n={train_stats['n']} | val_n={val_stats['n']} | time={elapsed:.0f}s"
        )

        torch.save(
            {
                "model": ranker.state_dict(),
                "scene_descriptor": scene.state_dict(),
            },
            CHECKPOINT_OUT / f"ranker_epoch_{epoch}.pt",
        )

        if val_stats["loss"] <= best_val:
            best_val = val_stats["loss"]
            torch.save(
                {
                    "model": ranker.state_dict(),
                    "scene_descriptor": scene.state_dict(),
                },
                CHECKPOINT_OUT / "ranker_best.pt",
            )

        torch.save(
            {
                "epoch": epoch,
                "model": ranker.state_dict(),
                "scene_descriptor": scene.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val,
                "history": history,
            },
            CHECKPOINT_OUT / "ranker_latest.pt",
        )

        (LOG_DIR / "training_curves.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    summary = {
        "checkpoint_in": str(CHECKPOINT_IN),
        "checkpoint_out": str(CHECKPOINT_OUT / "ranker_best.pt"),
        "cache_dir": str(CACHE_DIR),
        "epochs": epochs,
        "best_val_loss": best_val,
        "train_indices": len(train_indices),
        "val_indices": len(val_indices),
        "task_counts": {str(k): len(v) for k, v in task_buckets.items()},
    }
    (LOG_DIR / "competition_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Best checkpoint: {CHECKPOINT_OUT / 'ranker_best.pt'}")


if __name__ == "__main__":
    main()