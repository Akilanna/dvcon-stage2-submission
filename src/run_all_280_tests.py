#!/usr/bin/env python3
"""Run 20 images x 14 tasks and save all 280 tested outputs to a new folder.

For each image-task pair, this script always writes one output image.
Bounding box is drawn only when a strict task match is found.
"""

import csv
import json
import os
import subprocess
import time
from datetime import datetime

IMAGES = [
    "data/coco/train2014/COCO_train2014_000000000009.jpg",
    "data/coco/train2014/COCO_train2014_000000000025.jpg",
    "data/coco/train2014/COCO_train2014_000000000030.jpg",
    "data/coco/train2014/COCO_train2014_000000000034.jpg",
    "data/coco/train2014/COCO_train2014_000000000036.jpg",
    "data/coco/train2014/COCO_train2014_000000000049.jpg",
    "data/coco/train2014/COCO_train2014_000000000061.jpg",
    "data/coco/train2014/COCO_train2014_000000000071.jpg",
    "data/coco/train2014/COCO_train2014_000000000077.jpg",
    "data/coco/train2014/COCO_train2014_000000000081.jpg",
    "data/coco/train2014/COCO_train2014_000000000086.jpg",
    "data/coco/train2014/COCO_train2014_000000000089.jpg",
    "data/coco/train2014/COCO_train2014_000000000092.jpg",
    "data/coco/train2014/COCO_train2014_000000000094.jpg",
    "data/coco/train2014/COCO_train2014_000000000109.jpg",
    "data/coco/train2014/COCO_train2014_000000000110.jpg",
    "data/coco/train2014/COCO_train2014_000000000113.jpg",
    "data/coco/train2014/COCO_train2014_000000000127.jpg",
    "data/coco/train2014/COCO_train2014_000000000138.jpg",
    "data/coco/train2014/COCO_train2014_000000000142.jpg",
]

TASKS = [
    (0, "step_on_something"),
    (1, "sit_comfortably"),
    (2, "place_flowers"),
    (3, "get_potatoes_out_of_fire"),
    (4, "water_plant"),
    (5, "get_lemon_out_of_tea"),
    (6, "dig_hole"),
    (7, "open_bottle_of_beer"),
    (8, "open_parcel"),
    (9, "serve_wine"),
    (10, "pour_sugar"),
    (11, "smear_butter"),
    (12, "extinguish_fire"),
    (13, "pound_carpet"),
]


def parse_verification_stdout(output_text):
    """Extract match status and best-object metadata from verify_model.py logs."""
    selected_line = None
    low_or_incompatible = False
    no_detected = False

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if line.startswith("[!") and "Selection flagged" in line:
            low_or_incompatible = True
        if "No objects detected." in line:
            no_detected = True
        if line.startswith("[") and "]" in line and "det_conf=" in line:
            selected_line = line

    # Match convention for this run:
    # - no flagged warning
    # - and at least one detected object
    # - and verification succeeded
    is_match = (not low_or_incompatible) and (not no_detected)

    return {
        "is_match": is_match,
        "low_or_incompatible": low_or_incompatible,
        "no_detected": no_detected,
        "last_detected_line": selected_line,
    }


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("final_verification_results", f"blindspot_280_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    total_expected = len(IMAGES) * len(TASKS)
    total = 0
    ok = 0
    fail = 0
    match_count = 0

    manifest_rows = []
    start_time = time.time()

    print("=" * 78)
    print("BLIND-SPOT VERIFICATION RUN")
    print(f"Plan: {len(IMAGES)} images x {len(TASKS)} tasks = {total_expected} tests")
    print("Behavior: Save all 280 test images; draw bbox only when match is true")
    print(f"Output folder: {output_dir}")
    print("=" * 78)

    for img_index, image_path in enumerate(IMAGES, start=1):
        if not os.path.isfile(image_path):
            print(f"[SKIP] Missing image: {image_path}")
            continue

        image_base = os.path.splitext(os.path.basename(image_path))[0]

        for task_id, task_name in TASKS:
            total += 1
            output_png = os.path.join(output_dir, f"{image_base}_task{task_id:02d}_{task_name}.png")

            cmd = [
                ".venv/Scripts/python.exe",
                "verify_model.py",
                "--image",
                image_path,
                "--task",
                str(task_id),
                "--output",
                output_png,
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            parsed = parse_verification_stdout(proc.stdout)

            if proc.returncode == 0:
                ok += 1
            else:
                fail += 1

            if parsed["is_match"] and proc.returncode == 0:
                match_count += 1

            manifest_rows.append(
                {
                    "image": image_path,
                    "task_id": task_id,
                    "task_name": task_name,
                    "output_file": output_png,
                    "return_code": proc.returncode,
                    "is_match": int(parsed["is_match"] and proc.returncode == 0),
                    "low_or_incompatible": int(parsed["low_or_incompatible"]),
                    "no_detected": int(parsed["no_detected"]),
                }
            )

            elapsed = time.time() - start_time
            eta_seconds = (elapsed / total) * (total_expected - total) if total > 0 else 0
            status = "OK" if proc.returncode == 0 else "FAIL"
            print(
                f"[{total:3d}/{total_expected}] "
                f"img {img_index:02d}/{len(IMAGES)} "
                f"task {task_id:02d}({task_name:11s}) "
                f"{status} match={int(parsed['is_match'] and proc.returncode == 0)} "
                f"ETA={eta_seconds/60:5.1f}m"
            )

    manifest_csv = os.path.join(output_dir, "manifest_280.csv")
    manifest_json = os.path.join(output_dir, "manifest_280.json")
    summary_txt = os.path.join(output_dir, "SUMMARY_280.txt")

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "image",
                "task_id",
                "task_name",
                "output_file",
                "return_code",
                "is_match",
                "low_or_incompatible",
                "no_detected",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(manifest_json, "w", encoding="utf-8") as f_json:
        json.dump(
            {
                "run_id": run_id,
                "output_dir": output_dir,
                "expected_tests": total_expected,
                "actual_tests": total,
                "ok": ok,
                "fail": fail,
                "match_count": match_count,
                "non_match_count": max(total - match_count, 0),
                "rows": manifest_rows,
            },
            f_json,
            indent=2,
        )

    duration = time.time() - start_time
    with open(summary_txt, "w", encoding="utf-8") as f_summary:
        f_summary.write("=" * 78 + "\n")
        f_summary.write("BLIND-SPOT VERIFICATION SUMMARY\n")
        f_summary.write("=" * 78 + "\n")
        f_summary.write(f"Run ID: {run_id}\n")
        f_summary.write(f"Output folder: {output_dir}\n")
        f_summary.write(f"Expected tests: {total_expected}\n")
        f_summary.write(f"Completed tests: {total}\n")
        f_summary.write(f"Successful runs: {ok}\n")
        f_summary.write(f"Failed runs: {fail}\n")
        f_summary.write(f"Match count (bbox drawn): {match_count}\n")
        f_summary.write(f"Non-match count (no bbox): {max(total - match_count, 0)}\n")
        f_summary.write(f"Duration minutes: {duration/60:.2f}\n")
        f_summary.write("\n")
        f_summary.write("Files:\n")
        f_summary.write(f"- {manifest_csv}\n")
        f_summary.write(f"- {manifest_json}\n")
        f_summary.write(f"- {summary_txt}\n")

    print("\n" + "=" * 78)
    print("RUN COMPLETE")
    print(f"Output folder: {output_dir}")
    print(f"Tests completed: {total}/{total_expected}")
    print(f"Successful runs: {ok}, Failed runs: {fail}")
    print(f"Matches (bbox drawn): {match_count}")
    print(f"Non-matches (no bbox): {max(total - match_count, 0)}")
    print(f"Summary: {summary_txt}")
    print("=" * 78)


if __name__ == "__main__":
    main()
