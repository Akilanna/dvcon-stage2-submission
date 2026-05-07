#!/usr/bin/env python3
"""Generate a 14-task final demo grid from the verification outputs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "competition" / "full_demo_verification.csv"
OUT_PATH = ROOT / "competition" / "final_demo_grid.png"


def _load_rows():
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = _load_rows()
    ranked_by_task = {}
    for row in rows:
        task_id = int(row["task_id"])
        ranked_by_task.setdefault(task_id, []).append(row)

    for task_id in ranked_by_task:
        ranked_by_task[task_id].sort(
            key=lambda r: (
                r["rejected"].lower() == "true",
                -float(r["confidence"]),
            )
        )

    best_by_task = {}
    used_images = set()
    for task_id in range(14):
        candidates = ranked_by_task.get(task_id, [])
        chosen = None
        for row in candidates:
            if row["image_path"] not in used_images:
                chosen = row
                break
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is not None:
            best_by_task[task_id] = chosen
            used_images.add(chosen["image_path"])

    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")

    for task_id in range(14):
        ax = axes[task_id]
        row = best_by_task.get(task_id)
        if row is None:
            ax.text(0.5, 0.5, f"Task {task_id}\nmissing", ha="center", va="center")
            continue
        img = Image.open(ROOT / row["output_png"])
        ax.imshow(img)
        title = (
            f"Task {task_id}\n"
            f"pred: {row['prediction']}\n"
            f"conf: {float(row['confidence']):.3f} | rejected: {row['rejected']}"
        )
        ax.set_title(title, fontsize=10)

    for idx in range(14, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Competition Demo Grid", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PATH, dpi=160)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()