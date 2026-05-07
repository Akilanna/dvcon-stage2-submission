"""
Loss Functions for Object-Task Ranking
---------------------------------------
Focal Loss for binary suitability scoring.

Handles the severe class imbalance in COCO-Tasks where most detected
objects are non-preferred (label 0) and only ~1 per image is preferred.
"""

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> tuple:
    """
    Binary focal loss operating on raw logits (numerically stable).

    Uses BCEWithLogitsLoss internally so sigmoid + log are fused.

    Args:
        logits: [N] raw MLP outputs (before sigmoid).
        labels: [N] ground truth, 1.0 = preferred, 0.0 = not.
        alpha:  Weight for positive class.
        gamma:  Focusing parameter.

    Returns:
        (loss, metrics_dict)
    """
    n = logits.shape[0]
    p = torch.sigmoid(logits) if n > 0 else logits

    metrics = {
        "num_objects": n,
        "num_positive": int(labels.sum().item()) if n > 0 else 0,
        "avg_score": p.mean().item() if n > 0 else 0.0,
    }

    if n == 0:
        return torch.tensor(0.0, requires_grad=True), metrics

    # Numerically stable BCE (fuses sigmoid + log)
    ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    # p_t = probability of the true class
    p_t = p * labels + (1.0 - p) * (1.0 - labels)

    # Focal weighting
    alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    focal_weight = alpha_t * (1.0 - p_t).pow(gamma)
    loss = (focal_weight * ce).mean()

    return loss, metrics


if __name__ == "__main__":
    # Quick sanity check — logits (not probabilities)
    logits = torch.tensor([1.4, -0.8, -2.2, 0.4, -1.4])  # ~sigmoid → [0.8, 0.3, 0.1, 0.6, 0.2]
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])

    loss, m = focal_loss(logits, labels)
    print(f"loss:    {loss.item():.4f}")
    print(f"metrics: {m}")

    # Edge: empty
    loss0, m0 = focal_loss(torch.zeros(0), torch.zeros(0))
    print(f"empty loss: {loss0.item():.4f}")
