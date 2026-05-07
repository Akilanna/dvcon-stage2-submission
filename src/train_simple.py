""" Simple Training Script for COCO-Tasks Ranking Model
This script trains the RankingMLP model on cached features.
Run it manually in your terminal: py training\train_simple.py
"""
import json
import os
import sys
import time
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.feature_module.scene_descriptor import SceneDescriptor
from models.task_encoder.task_embedding import TaskEmbeddingManager
from models.ranker.ranking_mlp import RankingMLP, build_ranking_input
from training.cached_dataset import CachedFeatureDataset, cached_collate_fn
from training.loss_functions import focal_loss


def main():
    print("=" * 60)
    print("  COCO-Tasks Ranking Model Training")
    print("=" * 60)
    print()

    # Configuration
    CACHE_DIR = "data/cached_features"
    EMBEDDING_FILE = "data/embeddings/tasks.npy"
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    EPOCHS = 30
    LR = 1e-4
    VAL_SPLIT = 0.15

    print(f"Configuration:")
    print(f"  Cache dir: {CACHE_DIR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"  Validation split: {VAL_SPLIT:.0%}")
    print()

    # Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (no GPU detected)")
    print()

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    full_dataset = CachedFeatureDataset(CACHE_DIR)
    n_total = len(full_dataset)
    print(f"  Total samples: {n_total}")

    if n_total == 0:
        print("ERROR: No cached features found!")
        print("Run: py scripts/precompute_features.py")
        return

    # Split train/val
    n_val = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {n_train} | Val: {n_val}")
    print()

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=cached_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=cached_collate_fn)

    # Initialize models
    scene_descriptor = SceneDescriptor(use_gating=True)
    task_manager = TaskEmbeddingManager(EMBEDDING_FILE)
    ranker = RankingMLP(input_dim=731, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(
        list(ranker.parameters()) + list(scene_descriptor.parameters()),
        lr=LR
    )

    # Resume from checkpoint if exists
    latest_path = os.path.join(CHECKPOINT_DIR, "ranker_latest.pt")
    best_path = os.path.join(CHECKPOINT_DIR, "ranker_best.pt")
    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if os.path.isfile(latest_path):
        print(f"Resuming from: {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        ranker.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scene_descriptor" in ckpt:
            scene_descriptor.load_state_dict(ckpt["scene_descriptor"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch}")
        print(f"  Best val_loss so far: {best_val_loss:.4f}")
    else:
        print("Starting fresh training")
    print()

    # Training loop
    print("Starting training...")
    print("-" * 80)

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_t0 = time.time()

        # Training
        ranker.train()
        scene_descriptor.train()
        train_loss = 0.0
        train_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            sample = batch[0]
            features = sample["features"]
            class_ids = sample["class_ids"]
            labels = sample["labels"]
            task_id = sample["task_id"]

            task_embed = task_manager.get_embedding(task_id).to(features.device)
            scene = scene_descriptor.compute(features, task_embed)
            ranking_input = build_ranking_input(features, scene, task_embed, class_ids)

            logits = ranker(ranking_input.to(device))
            loss, _ = focal_loss(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_samples += 1

        train_loss /= train_samples

        # Validation
        ranker.eval()
        scene_descriptor.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                sample = batch[0]
                features = sample["features"]
                class_ids = sample["class_ids"]
                labels = sample["labels"]
                task_id = sample["task_id"]

                task_embed = task_manager.get_embedding(task_id).to(features.device)
                scene = scene_descriptor.compute(features, task_embed)
                ranking_input = build_ranking_input(features, scene, task_embed, class_ids)

                logits = ranker(ranking_input.to(device))
                loss, _ = focal_loss(logits, labels.to(device))

                val_loss += loss.item()
                val_samples += 1

        val_loss /= val_samples

        epoch_time = time.time() - epoch_t0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"time: {epoch_time:.0f}s | "
              f"lr: {current_lr:.6f}")

        # Save checkpoint
        torch.save({
            "model": ranker.state_dict(),
            "scene_descriptor": scene_descriptor.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, f"ranker_epoch_{epoch}.pt"))

        # Save latest (for resume)
        torch.save({
            "epoch": epoch,
            "model": ranker.state_dict(),
            "scene_descriptor": scene_descriptor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        }, latest_path)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": ranker.state_dict(),
                "scene_descriptor": scene_descriptor.state_dict(),
            }, best_path)
            print(f"  >> NEW BEST! val_loss: {best_val_loss:.4f}")

        # Update history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time": epoch_time,
        })

        # Save training curves
        curves_path = os.path.join(LOG_DIR, "training_curves.json")
        with open(curves_path, "w") as f:
            json.dump(history, f, indent=2)

    print("-" * 80)
    print(f"Training complete!")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()