"""
Pipeline Evaluation Script
---------------------------
Runs the full trained pipeline on the test set and reports results.

Responsibilities:
- Load trained model checkpoint
- Run end-to-end inference on test split: detect -> extract -> embed -> rank
- Compute all evaluation metrics
- Generate per-task and aggregate result tables
- Optionally produce qualitative visualizations

Usage: python evaluation/evaluate_pipeline.py --checkpoint checkpoints/ranker_best.pt
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from models.feature_module.feature_extractor import FeatureExtractor
from models.feature_module.scene_descriptor import SceneDescriptor
from models.task_encoder.task_embedding import TaskEmbeddingManager
from models.ranker.ranking_mlp import RankingMLP, build_ranking_input
from training.dataset_loader import COCOTasksDataset, collate_fn
from evaluation.metrics import compute_metrics, print_metrics_table


def evaluate(
    checkpoint: str = "checkpoints/ranker_best.pt",
    projector_ckpt: str = "checkpoints/projector.pt",
    annotation_dir: str = "data/annotations",
    image_root: str = "data/coco/val2014",
    embedding_file: str = "data/embeddings/tasks.npy",
    device: str = "cpu",
    max_samples: int = 0,
    output_json: str = "",
) -> dict:
    """
    Run evaluation on the COCO-Tasks test split using ground-truth boxes.

    Ground-truth boxes are used (not YOLO detections) to isolate the ranking
    quality from detection quality, consistent with the training setup.

    Args:
        checkpoint:      Path to RankingMLP state dict (.pt).
        projector_ckpt:  Path to ResNet18 projector weights.
        annotation_dir:  Directory with task_{1-14}_test.json files.
        image_root:      Directory with COCO val2014 images.
        embedding_file:  Path to precomputed task embeddings [14, 384].
        device:          'cpu' or 'cuda'.
        max_samples:     Limit evaluation to first N samples (0 = unlimited).
        output_json:     If set, write metrics dict to this path.

    Returns:
        Metrics dict (see evaluation/metrics.py).
    """
    print(f"Checkpoint  : {checkpoint}")
    print(f"Image root  : {image_root}")
    print(f"Device      : {device}")

    # ---- Models ----
    feature_extractor = FeatureExtractor(feature_dim=128, device=device)
    feature_extractor.eval()

    if os.path.isfile(projector_ckpt):
        proj_sd = torch.load(projector_ckpt, map_location=device, weights_only=True)
        feature_extractor.projector.load_state_dict(proj_sd)
        print(f"Projector   : loaded from {projector_ckpt}")
    else:
        print(f"WARNING: projector checkpoint not found at {projector_ckpt}")

    task_manager = TaskEmbeddingManager(embedding_file)

    ranker = RankingMLP(input_dim=731, hidden_dim=128).to(device)
    sd = torch.load(checkpoint, map_location=device, weights_only=False)
    use_gating = False
    scene_descriptor_sd = None
    if isinstance(sd, dict) and "fc1.weight" in sd:
        ranker_sd = sd
    elif isinstance(sd, dict) and "model" in sd:
        ranker_sd = sd["model"]
        if "scene_descriptor" in sd:
            use_gating = True
            scene_descriptor_sd = sd["scene_descriptor"]
    else:
        raise RuntimeError(f"Unrecognised checkpoint format in {checkpoint}: {list(sd.keys())}")

    scene_descriptor = SceneDescriptor(use_gating=use_gating).to(device)
    if scene_descriptor_sd is not None:
        scene_descriptor.load_state_dict(scene_descriptor_sd)

    ranker.load_state_dict(ranker_sd)
    ranker.eval()
    scene_descriptor.eval()
    print(f"Ranker      : loaded from {checkpoint}")
    print(f"Scene desc. : {'task-gated' if use_gating else 'mean-pooling'}")

    # ---- Dataset ----
    dataset = COCOTasksDataset(annotation_dir, image_root, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    n_total = len(dataset)
    print(f"Test samples: {n_total}")

    # ---- Inference ----
    scores_list = []
    labels_list = []
    task_ids_list = []

    t0 = time.time()
    n_done = 0
    n_skipped = 0

    with torch.no_grad():
        for batch in loader:
            if max_samples > 0 and n_done >= max_samples:
                break

            sample = batch[0]
            image     = sample["image"]
            boxes     = sample["boxes"]
            class_ids = sample["class_ids"]
            labels    = sample["labels"]
            task_id   = sample["task_id"]

            if boxes.shape[0] == 0:
                n_skipped += 1
                continue

            features      = feature_extractor(image, boxes)          # [N, 128]
            task_embed    = task_manager.get_embedding(task_id).to(features.device)       # [384]
            scene         = scene_descriptor.compute(features, task_embed)  # [128]
            ranking_input = build_ranking_input(
                features, scene, task_embed, class_ids
            )                                                         # [N, 731]
            logits = ranker(ranking_input.to(device))                 # [N]

            scores_list.append(logits.cpu())
            labels_list.append(labels.cpu())
            task_ids_list.append(task_id)
            n_done += 1

            if n_done % 500 == 0:
                elapsed = time.time() - t0
                print(f"  [{n_done}/{n_total}] elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nInference done: {n_done} samples in {elapsed:.1f}s "
          f"({n_skipped} skipped, no objects)")

    # ---- Metrics ----
    metrics = compute_metrics(scores_list, labels_list, task_ids=task_ids_list)
    print_metrics_table(metrics)

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {output_json}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ranking pipeline on test set")
    parser.add_argument("--checkpoint",  default="checkpoints/ranker_best.pt")
    parser.add_argument("--projector",   default="checkpoints/projector.pt")
    parser.add_argument("--ann-dir",     default="data/annotations")
    parser.add_argument("--image-root",  default="data/coco/val2014",
                        help="COCO val2014 image directory")
    parser.add_argument("--embeddings",  default="data/embeddings/tasks.npy")
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit to first N samples (0=all)")
    parser.add_argument("--output-json", default="logs/eval_results.json",
                        help="Write metrics to this JSON file")
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        projector_ckpt=args.projector,
        annotation_dir=args.ann_dir,
        image_root=args.image_root,
        embedding_file=args.embeddings,
        device=args.device,
        max_samples=args.max_samples,
        output_json=args.output_json,
    )

