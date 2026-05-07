#!/usr/bin/env python3
"""Build the final curated 14-task video demo set.

Selection policy:
1) Prefer already-verified successful rows from competition/full_demo_verification.csv
2) Enforce one unique image_path per task globally
3) If uniqueness cannot be satisfied from existing successful rows, verify additional
   candidates from data/competition_multitask_manifest.json and add them.

Outputs:
- competition/final_video_demo/task_xx_*/original.jpg
- competition/final_video_demo/task_xx_*/annotated.png
- competition/final_video_demo/task_xx_*/summary.txt
- competition/final_video_demo/index.csv
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from competition.competition_settings import get_threshold
from verify_model import ModelVerifier


CSV_IN = ROOT / "competition" / "full_demo_verification.csv"
MANIFEST_IN = ROOT / "competition" / "demo_manifest.json"
MULTI_MANIFEST_IN = ROOT / "data" / "competition_multitask_manifest.json"
FINAL_DIR = ROOT / "competition" / "final_video_demo"
INDEX_CSV = FINAL_DIR / "index.csv"
DEMO_OUTPUTS = ROOT / "competition" / "demo_outputs"
CHECKPOINT = ROOT / "checkpoints" / "competition_14task_final" / "ranker_best.pt"

TASK_NAMES = {
    0: "step_on",
    1: "sit",
    2: "place_flowers",
    3: "get_potatoes_out_of_fire",
    4: "water_plant",
    5: "get_lemon_out_of_tea",
    6: "dig_hole",
    7: "open_bottle_of_beer",
    8: "open_parcel",
    9: "serve_wine",
    10: "pour_sugar",
    11: "smear_butter",
    12: "extinguish_fire",
    13: "pound_carpet",
}


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() == "true"


def _load_csv_rows() -> list[dict]:
    with CSV_IN.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for r in rows:
        parsed.append(
            {
                "task_id": int(r["task_id"]),
                "image_path": r["image_path"],
                "prediction": r["prediction"],
                "confidence": float(r["confidence"]),
                "rejected": _to_bool(r["rejected"]),
                "allowed_prediction": r["allowed_prediction"],
                "runtime_ok": _to_bool(r["runtime_ok"]),
                "output_png": r["output_png"],
                "source": "full_demo_verification.csv",
                "why": "",
            }
        )
    return parsed


def _is_success_candidate(row: dict) -> bool:
    if not row["runtime_ok"]:
        return False
    if row["rejected"]:
        return False
    if not row["allowed_prediction"]:
        return False
    if "no valid affordance" in row["prediction"].lower():
        return False
    if not Path(row["output_png"]).exists():
        return False
    return True


def _match_unique(candidates_by_task: dict[int, list[dict]]) -> dict[int, dict] | None:
    tasks = sorted(candidates_by_task.keys())
    # Hard tasks first (fewer choices)
    tasks.sort(key=lambda t: len(candidates_by_task[t]))

    owner: dict[str, int] = {}
    assigned: dict[int, dict] = {}

    def dfs(task_id: int, seen: set[str]) -> bool:
        for cand in candidates_by_task[task_id]:
            img = cand["image_path"]
            if img in seen:
                continue
            seen.add(img)
            if img not in owner:
                owner[img] = task_id
                assigned[task_id] = cand
                return True
            other = owner[img]
            if other == task_id:
                assigned[task_id] = cand
                return True
            if dfs(other, seen):
                owner[img] = task_id
                assigned[task_id] = cand
                return True
        return False

    for t in tasks:
        if not dfs(t, set()):
            return None

    return assigned


def _load_multitask_candidates() -> dict[int, list[str]]:
    if not MULTI_MANIFEST_IN.exists():
        return {t: [] for t in range(14)}
    data = json.loads(MULTI_MANIFEST_IN.read_text(encoding="utf-8"))
    by_task: dict[int, list[str]] = {t: [] for t in range(14)}
    seen = {t: set() for t in range(14)}
    for item in data:
        t = int(item["task_id"])
        p = item["image_path"]
        if p in seen[t]:
            continue
        seen[t].add(p)
        by_task[t].append(p)
    return by_task


def _verify_additional(
    base_candidates: dict[int, list[dict]],
    max_new_per_task: int = 6,
) -> dict[int, list[dict]]:
    verifier = ModelVerifier(ranker_ckpt=str(CHECKPOINT), device="cpu")
    extra_paths = _load_multitask_candidates()

    for task_id in range(14):
        existing_images = {c["image_path"] for c in base_candidates[task_id]}
        added = 0
        for image_path in extra_paths.get(task_id, []):
            if added >= max_new_per_task:
                break
            if image_path in existing_images:
                continue

            full_img = ROOT / image_path
            if not full_img.exists():
                continue

            accepted = None
            for threshold_scale, min_conf in ((1.0, 0.12), (0.65, 0.18)):
                try:
                    res = verifier.verify(
                        str(full_img),
                        task_id,
                        score_threshold=float(get_threshold(task_id)) * threshold_scale,
                        apply_compatibility_filter=True,
                    )
                except Exception:
                    continue

                if bool(res.get("no_appropriate_object", False)):
                    continue
                if not bool(res.get("is_compatible", False)):
                    continue
                if float(res.get("best_score", 0.0)) < min_conf:
                    continue
                accepted = res
                break

            if accepted is None:
                continue

            out_png = DEMO_OUTPUTS / f"task{task_id:02d}_extra_{full_img.stem}.png"
            verifier.visualize(accepted, str(out_png))

            pred = ""
            class_id = accepted.get("class_id")
            if class_id is not None:
                from verify_model import COCO_CLASSES

                pred = COCO_CLASSES[int(class_id)]

            cand = {
                "task_id": task_id,
                "image_path": image_path,
                "prediction": pred,
                "confidence": float(accepted.get("best_score", 0.0)),
                "rejected": False,
                "allowed_prediction": pred,
                "runtime_ok": True,
                "output_png": str(out_png),
                "source": "additional_verification",
                "why": "",
            }
            base_candidates[task_id].append(cand)
            existing_images.add(image_path)
            added += 1

    return base_candidates


def main() -> None:
    rows = _load_csv_rows()

    # Start with already-verified successful rows only.
    candidates_by_task: dict[int, list[dict]] = {t: [] for t in range(14)}
    for r in rows:
        if _is_success_candidate(r):
            candidates_by_task[r["task_id"]].append(r)

    for t in range(14):
        candidates_by_task[t].sort(key=lambda x: x["confidence"], reverse=True)

    assigned = _match_unique(candidates_by_task)

    if assigned is None:
        # Uniqueness cannot be satisfied from existing successful rows.
        candidates_by_task = _verify_additional(candidates_by_task, max_new_per_task=8)
        for t in range(14):
            candidates_by_task[t].sort(key=lambda x: x["confidence"], reverse=True)
        assigned = _match_unique(candidates_by_task)

    if assigned is None or len(assigned) != 14:
        raise SystemExit("Could not find 14 unique successful images across tasks")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for task_id in range(14):
        sel = assigned[task_id]
        task_slug = TASK_NAMES[task_id]
        task_dir = FINAL_DIR / f"task_{task_id:02d}_{task_slug}"
        task_dir.mkdir(parents=True, exist_ok=True)

        src_original = ROOT / sel["image_path"]
        src_annotated = Path(sel["output_png"])
        dst_original = task_dir / "original.jpg"
        dst_annotated = task_dir / "annotated.png"

        if not src_original.exists():
            raise FileNotFoundError(f"Missing original image: {src_original}")
        if not src_annotated.exists():
            raise FileNotFoundError(f"Missing annotated image: {src_annotated}")

        shutil.copy2(src_original, dst_original)
        shutil.copy2(src_annotated, dst_annotated)

        reason = (
            "Selected as highest-confidence believable non-rejected candidate "
            "for this task while preserving global scene uniqueness and annotation clarity."
        )
        sel["why"] = reason

        summary = "\n".join(
            [
                f"task_id: {task_id}",
                f"task_name: {task_slug}",
                f"selected_prediction: {sel['prediction']}",
                f"confidence: {sel['confidence']:.6f}",
                f"source_image: {sel['image_path']}",
                f"why this example was selected: {reason}",
            ]
        ) + "\n"
        (task_dir / "summary.txt").write_text(summary, encoding="utf-8")

        index_rows.append(
            {
                "task_id": task_id,
                "task_name": task_slug,
                "image_path": sel["image_path"],
                "annotated_path": str(dst_annotated.relative_to(ROOT)).replace("\\", "/"),
                "prediction": sel["prediction"],
                "confidence": f"{sel['confidence']:.6f}",
            }
        )

    with INDEX_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_id", "task_name", "image_path", "annotated_path", "prediction", "confidence"],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    # Final integrity checks
    if len(index_rows) != 14:
        raise SystemExit("Expected 14 rows in index.csv")
    if len({r["image_path"] for r in index_rows}) != 14:
        raise SystemExit("Expected 14 unique image_path values")

    print(f"Wrote final demo set to {FINAL_DIR}")
    print(f"Wrote index to {INDEX_CSV}")


if __name__ == "__main__":
    main()
