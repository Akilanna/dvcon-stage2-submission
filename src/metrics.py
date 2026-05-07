"""
Evaluation Metrics
------------------
Computes performance metrics for the object selection pipeline.

Responsibilities:
- Top-1 accuracy: fraction of images where the highest-scored object is correct
- Top-3 accuracy: fraction where correct object is in top-3 scored
- Mean reciprocal rank (MRR) of the correct object
- Per-task breakdown of accuracy across all 14 COCO-Tasks

Input:  Predicted scores per object, ground-truth labels
Output: Dictionary of metric names to values
"""

from typing import Dict, List, Optional

import numpy as np
import torch

NUM_TASKS = 14

TASK_NAMES = [
    "step on something",
    "sit comfortably",
    "place flowers",
    "get potatoes out of fire",
    "water plant",
    "get lemon out of tea",
    "dig hole",
    "open bottle of beer",
    "open parcel",
    "serve wine",
    "pour sugar",
    "smear butter",
    "extinguish fire",
    "pound carpet",
]


def compute_metrics(
    scores_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    task_ids: Optional[List[int]] = None,
    num_tasks: int = NUM_TASKS,
) -> Dict[str, float]:
    """
    Compute Top-1, Top-3, and MRR metrics over a collection of ranked examples.

    Each element of scores_list / labels_list corresponds to one image+task pair,
    where scores_list[i] is a vector of raw scores per detected object and
    labels_list[i] is a binary vector (1 = preferred for the task, 0 = not).

    Args:
        scores_list: List of [N_i] score tensors (higher = more relevant).
        labels_list: List of [N_i] binary label tensors (1=preferred, 0=not).
        task_ids:    Optional list of task IDs (int 0-13) parallel to scores_list.
        num_tasks:   Number of tasks for per-task breakdown.

    Returns:
        Dict with keys:
            top1, top3, mrr, n_samples
            (if task_ids provided) task_{t}_top1, task_{t}_mrr, task_{t}_n for each t
    """
    top1_hits: List[float] = []
    top3_hits: List[float] = []
    mrr_values: List[float] = []

    per_task_top1: Dict[int, List[float]] = {t: [] for t in range(num_tasks)}
    per_task_mrr:  Dict[int, List[float]] = {t: [] for t in range(num_tasks)}

    for i, (scores, labels) in enumerate(zip(scores_list, labels_list)):
        scores = scores.float().cpu()
        labels = labels.float().cpu()

        if labels.sum() == 0:
            continue  # skip images with no positive annotation

        # Rank objects by descending score
        order = torch.argsort(scores, descending=True)
        sorted_labels = labels[order]

        # Top-1: highest-ranked object is a positive
        top1 = float(sorted_labels[0].item() > 0)
        top1_hits.append(top1)

        # Top-3: at least one positive in the top-3
        top3 = float(sorted_labels[:3].sum().item() > 0)
        top3_hits.append(top3)

        # MRR: 1 / rank_of_first_positive (rank is 1-indexed)
        pos_positions = (sorted_labels > 0).nonzero(as_tuple=True)[0]
        first_rank = int(pos_positions[0].item()) + 1
        mrr_values.append(1.0 / first_rank)

        if task_ids is not None:
            tid = task_ids[i]
            if 0 <= tid < num_tasks:
                per_task_top1[tid].append(top1)
                per_task_mrr[tid].append(1.0 / first_rank)

    n = len(top1_hits)
    result: Dict[str, float] = {
        "top1":      float(np.mean(top1_hits))  if n > 0 else 0.0,
        "top3":      float(np.mean(top3_hits))  if n > 0 else 0.0,
        "mrr":       float(np.mean(mrr_values)) if n > 0 else 0.0,
        "n_samples": float(n),
    }

    if task_ids is not None:
        for t in range(num_tasks):
            v1 = per_task_top1[t]
            vm = per_task_mrr[t]
            result[f"task_{t}_top1"] = float(np.mean(v1)) if v1 else float("nan")
            result[f"task_{t}_mrr"]  = float(np.mean(vm)) if vm else float("nan")
            result[f"task_{t}_n"]    = float(len(v1))

    return result


def print_metrics_table(metrics: Dict[str, float]) -> None:
    """Pretty-print aggregate and per-task metrics."""
    print("\n=== Aggregate Metrics ===")
    print(f"  Top-1 Accuracy : {metrics['top1']:.4f}  ({metrics['top1']*100:.1f}%)")
    print(f"  Top-3 Accuracy : {metrics['top3']:.4f}  ({metrics['top3']*100:.1f}%)")
    print(f"  MRR            : {metrics['mrr']:.4f}")
    print(f"  Samples        : {int(metrics['n_samples'])}")

    per_task_keys = [k for k in metrics if k.startswith("task_") and k.endswith("_top1")]
    if not per_task_keys:
        return

    print("\n=== Per-Task Breakdown ===")
    print(f"  {'ID':<4} {'Task':<30} {'Top-1':>7} {'MRR':>7} {'N':>6}")
    print("  " + "-" * 56)
    for t in range(NUM_TASKS):
        name = TASK_NAMES[t] if t < len(TASK_NAMES) else f"task_{t}"
        top1_t = metrics.get(f"task_{t}_top1", float("nan"))
        mrr_t  = metrics.get(f"task_{t}_mrr",  float("nan"))
        n_t    = int(metrics.get(f"task_{t}_n", 0))
        top1_s = f"{top1_t*100:.1f}%" if not np.isnan(top1_t) else "  n/a"
        mrr_s  = f"{mrr_t:.4f}"       if not np.isnan(mrr_t)  else "  n/a"
        print(f"  {t:<4} {name:<30} {top1_s:>7} {mrr_s:>7} {n_t:>6}")

