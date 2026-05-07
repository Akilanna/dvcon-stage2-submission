"""
Training Pipeline for Ranking Network
--------------------------------------
Trains only the RankingMLP using ground-truth boxes from COCO-Tasks.

YOLO is NOT used during training (frozen detector — proposal constraint).
FeatureExtractor runs in eval() mode (frozen backbone).
Only RankingMLP weights are updated.

Usage: python training/train_ranker.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, random_split

from models.feature_module.feature_extractor import FeatureExtractor
from models.feature_module.scene_descriptor import SceneDescriptor
from models.task_encoder.task_embedding import TaskEmbeddingManager
from models.ranker.ranking_mlp import RankingMLP, build_ranking_input
from training.dataset_loader import COCOTasksDataset, collate_fn
from training.loss_functions import focal_loss


def _run_epoch(loader, feature_extractor, scene_descriptor, task_manager,
               ranker, device, optimizer=None, diag=False):
    """Run one train or val epoch. If optimizer is None, validation mode."""
    is_train = optimizer is not None
    ranker.train() if is_train else ranker.eval()

    total_loss = 0.0
    total_objects = 0
    total_positives = 0
    num_samples = 0
    n_total = len(loader)
    epoch_start = time.time()

    for batch in loader:
        sample = batch[0]

        image = sample["image"]
        boxes = sample["boxes"]
        class_ids = sample["class_ids"]
        labels = sample["labels"]
        task_id = sample["task_id"]

        if boxes.shape[0] == 0:
            continue

        with torch.no_grad():
            features = feature_extractor(image, boxes)
            task_embed = task_manager.get_embedding(task_id).to(features.device)
            scene = scene_descriptor.compute(features, task_embed)
            ranking_input = build_ranking_input(
                features, scene, task_embed, class_ids
            )

        if diag and num_samples == 0:
            print(f"  feature mean: {features.mean().item():.4f}")
            print(f"  feature std:  {features.std().item():.4f}")

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

        if is_train and num_samples % 1000 == 0:
            elapsed = time.time() - epoch_start
            avg_loss = total_loss / num_samples
            avg_obj = total_objects / num_samples
            print(f"  [{num_samples}/{n_total}] "
                  f"loss={avg_loss:.4f} obj/img={avg_obj:.1f} "
                  f"elapsed={elapsed:.0f}s", flush=True)

    if num_samples == 0:
        return {"loss": 0.0, "obj_per_img": 0.0, "pos_per_img": 0.0, "n": 0}

    return {
        "loss": total_loss / num_samples,
        "obj_per_img": total_objects / num_samples,
        "pos_per_img": total_positives / num_samples,
        "n": num_samples,
    }


def train(
    annotation_dir: str = "data/annotations",
    image_root: str = "data/coco/train2014",
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

    # ---- Dataset integrity check ----
    if not os.path.isdir(image_root):
        print(f"ERROR: image_root not found: {image_root}")
        sys.exit(1)

    img_count = len([f for f in os.listdir(image_root) if f.endswith(".jpg")])
    print(f"Image directory: {image_root}")
    print(f"Image count: {img_count}")
    if img_count < 80_000:
        print(f"ERROR: Expected >= 80,000 images, found {img_count}. Aborting.")
        sys.exit(1)

    # ---- Data ----
    full_dataset = COCOTasksDataset(annotation_dir, image_root)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    if n_train < 1:
        # Tiny dataset: use same data for train and val
        train_ds = full_dataset
        val_ds = full_dataset
        n_train = n_total
        n_val = n_total
    else:
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ---- Dataset statistics ----
    print(f"\nDataset: {n_total} total | {n_train} train | {n_val} val")
    # Sample statistics from first 200 samples
    sample_n = min(200, n_total)
    total_obj, total_pos = 0, 0
    for i in range(sample_n):
        s = full_dataset[i]
        total_obj += len(s["labels"])
        total_pos += int(s["labels"].sum().item())
    avg_obj = total_obj / sample_n
    avg_pos = total_pos / sample_n
    print(f"avg objects/image: {avg_obj:.1f}")
    print(f"avg positives/image: {avg_pos:.1f}")
    print()

    # ---- Models ----
    feature_extractor = FeatureExtractor(feature_dim=128, device=device)
    feature_extractor.eval()

    scene_descriptor = SceneDescriptor()
    task_manager = TaskEmbeddingManager(embedding_file)

    ranker = RankingMLP(input_dim=731, hidden_dim=128).to(device)

    # ---- Optimizer (only ranking MLP) ----
    optimizer = torch.optim.Adam(ranker.parameters(), lr=lr)

    # ---- Resume from checkpoint if available ----
    best_val_loss = float("inf")
    history = []
    start_epoch = 1

    latest_path = os.path.join(checkpoint_dir, "ranker_latest.pt")
    if os.path.isfile(latest_path):
        print(f"Resuming from {latest_path}")
        ckpt = torch.load(latest_path, weights_only=False)
        ranker.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    else:
        print("Starting fresh training (no checkpoint found)")

    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.time()
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Train
        t = _run_epoch(
            train_loader, feature_extractor, scene_descriptor, task_manager,
            ranker, device, optimizer=optimizer, diag=(epoch == 1),
        )

        # Validate
        print("  validating...")
        with torch.no_grad():
            v = _run_epoch(
                val_loader, feature_extractor, scene_descriptor, task_manager,
                ranker, device, optimizer=None,
            )

        epoch_time = time.time() - epoch_t0
        print(
            f"Epoch {epoch:3d} | "
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
        })

        # Save every epoch
        ckpt_path = os.path.join(checkpoint_dir, f"ranker_epoch_{epoch}.pt")
        torch.save(ranker.state_dict(), ckpt_path)

        # Save resumable checkpoint (model + optimizer + state)
        latest_path = os.path.join(checkpoint_dir, "ranker_latest.pt")
        torch.save({
            "epoch": epoch,
            "model": ranker.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        }, latest_path)

        # Save best
        if v["loss"] < best_val_loss:
            best_val_loss = v["loss"]
            best_path = os.path.join(checkpoint_dir, "ranker_best.pt")
            torch.save(ranker.state_dict(), best_path)
            print(f"  >> new best val_loss: {best_val_loss:.4f}")

    # Save training curves
    curves_path = os.path.join(log_dir, "training_curves.json")
    with open(curves_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Curves saved to {curves_path}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    print("=== COCO-Tasks Ranker Training ===")
    train()
