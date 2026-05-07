"""
Fast Training on Cached Features
---------------------------------
Trains RankingMLP using pre-computed features (no CNN computation).
Scene descriptor and ranking input recomputed from cached features.
Each epoch takes ~2-3 minutes instead of ~3.5 hours.

Usage: py training/train_cached.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, random_split

from models.feature_module.scene_descriptor import SceneDescriptor
from models.task_encoder.task_embedding import TaskEmbeddingManager
from models.ranker.ranking_mlp import RankingMLP, build_ranking_input
from training.cached_dataset import CachedFeatureDataset, cached_collate_fn
from training.loss_functions import focal_loss


def _run_epoch(loader, scene_descriptor, task_manager,
               ranker, device, optimizer=None):
    """Run one train or val epoch on cached features."""
    is_train = optimizer is not None
    ranker.train() if is_train else ranker.eval()
    if hasattr(scene_descriptor, 'parameters'):
        scene_descriptor.train() if is_train else scene_descriptor.eval()

    total_loss = 0.0
    total_objects = 0
    total_positives = 0
    num_samples = 0

    for batch in loader:
        sample = batch[0]

        features = sample["features"]     # [N, 128]
        class_ids = sample["class_ids"]   # [N]
        labels = sample["labels"]         # [N]
        task_id = sample["task_id"]       # int

        if features.shape[0] == 0:
            continue

        # Recompute scene descriptor and task embedding from cached features
        task_embed = task_manager.get_embedding(task_id).to(features.device)
        scene = scene_descriptor.compute(features, task_embed)
        ranking_input = build_ranking_input(
            features, scene, task_embed, class_ids
        )

        logits = ranker(ranking_input.to(device))
        loss, metrics = focal_loss(logits, labels.to(device))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_objects += metrics["num_objects"]
        total_positives += metrics["num_positive"]
        num_samples += 1

    if num_samples == 0:
        return {"loss": 0.0, "obj_per_img": 0.0, "pos_per_img": 0.0, "n": 0}

    return {
        "loss": total_loss / num_samples,
        "obj_per_img": total_objects / num_samples,
        "pos_per_img": total_positives / num_samples,
        "n": num_samples,
    }


def train(
    cache_dir: str = "data/cached_features",
    embedding_file: str = "data/embeddings/tasks.npy",
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    epochs: int = 15,
    lr: float = 1e-4,
    val_split: float = 0.15,
    device: str = "cpu",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Device: {device}")

    # ---- Data ----
    full_dataset = CachedFeatureDataset(cache_dir)
    n_total = len(full_dataset)

    if n_total == 0:
        print("ERROR: No cached features found. Run scripts/precompute_features.py first.")
        sys.exit(1)

    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=cached_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=cached_collate_fn)
    # Each sample has a variable number of objects; batching would require padding,
    # so training intentionally runs one sample per step.
    assert train_loader.batch_size == 1

    print(f"Dataset: {n_total} cached | {n_train} train | {n_val} val")

    # ---- Model ----
    scene_descriptor = SceneDescriptor(use_gating=True)
    task_manager = TaskEmbeddingManager(embedding_file)
    assert task_manager.embedding_dim == 384, "Task embedding dimension must be 384"
    ranker = RankingMLP(input_dim=731, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(
        list(ranker.parameters()) + list(scene_descriptor.parameters()), lr=lr
    )

    # ---- Resume ----
    best_val_loss = float("inf")
    history = []
    start_epoch = 1

    latest_path = os.path.join(checkpoint_dir, "ranker_latest.pt")
    if os.path.isfile(latest_path):
        print(f"Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        ranker.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scene_descriptor" in ckpt:
            scene_descriptor.load_state_dict(ckpt["scene_descriptor"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    else:
        print("Starting fresh training")

    if start_epoch > epochs:
        print(f"Training already complete (start_epoch={start_epoch} > epochs={epochs}).")
        return

    # ---- Training loop ----
    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.time()

        t = _run_epoch(train_loader, scene_descriptor, task_manager,
                      ranker, device, optimizer=optimizer)

        with torch.no_grad():
            v = _run_epoch(val_loader, scene_descriptor, task_manager,
                          ranker, device, optimizer=None)

        epoch_time = time.time() - epoch_t0

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss: {t['loss']:.4f} | "
            f"val_loss: {v['loss']:.4f} | "
            f"obj/img: {t['obj_per_img']:.1f} | "
            f"pos/img: {t['pos_per_img']:.1f} | "
            f"time: {epoch_time:.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": t["loss"],
            "val_loss": v["loss"],
            "obj_per_img": t["obj_per_img"],
            "pos_per_img": t["pos_per_img"],
            "epoch_time": epoch_time,
        })

        # Save epoch checkpoint
        torch.save({
            "model": ranker.state_dict(),
            "scene_descriptor": scene_descriptor.state_dict(),
        }, os.path.join(checkpoint_dir, f"ranker_epoch_{epoch}.pt"))

        # Save best
        if v["loss"] < best_val_loss:
            best_val_loss = v["loss"]
            torch.save({
                "model": ranker.state_dict(),
                "scene_descriptor": scene_descriptor.state_dict(),
            }, os.path.join(checkpoint_dir, "ranker_best.pt"))
            print(f"  >> new best val_loss: {best_val_loss:.4f}")

        # Save resumable latest (after best_val_loss update)
        torch.save({
            "epoch": epoch,
            "model": ranker.state_dict(),
            "scene_descriptor": scene_descriptor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        }, latest_path)

        # Save curves incrementally
        curves_path = os.path.join(log_dir, "training_curves.json")
        with open(curves_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    print("=== COCO-Tasks Fast Cached Training ===")
    train()
