"""
Stage 2A Query Interface
------------------------
Provides a user-friendly interface for the 14 COCO-Tasks questions.
This script answers all 14 predefined questions from the Stage 2A requirements.

Usage:
  py stage2a_interface.py --image <path> --query "<question>"
  py stage2a_interface.py --image <path> --all  # Answer all 14 questions

The 14 Questions (COCO-Tasks):
  1. "What object should I use to cut something?"
  2. "What object should I use to sit on something?"
  3. "What object should I use to hold something (in hand)?"
  4. "What object should I use to make food?"
  5. "What object should I use to look through something?"
  6. "What object should I use to play music?"
  7. "What object should I use to read something?"
  8. "What object should I use to write something?"
  9. "What object should I use to scoop something?"
  10. "What object should I use to pound something?"
  11. "What object should I use to support something?"
  12. "What object should I use to wear something?"
  13. "What object should I use to carry/pick up something?"
  14. "What object should I use to open/close something?"
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.run_pipeline import TaskAwareObjectSelector

# COCO class names from Ultralytics YOLO (0-indexed, matches model.names)
# Verified against YOLOv8's built-in COCO class IDs (80 classes: 0-79)
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
    42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

# 14 COCO-Tasks with their natural language queries (from COCO-Tasks paper)
COCO_TASKS = [
    {"id": 0, "name": "step_on_something", "query": "What object should I use to step on something?", "description": "Step on something"},
    {"id": 1, "name": "sit_comfortably", "query": "What object should I use to sit comfortably?", "description": "Sit comfortably"},
    {"id": 2, "name": "place_flowers", "query": "What object should I use to place flowers?", "description": "Place flowers"},
    {"id": 3, "name": "get_potatoes_out_of_fire", "query": "What object should I use to get potatoes out of fire?", "description": "Get potatoes out of fire"},
    {"id": 4, "name": "water_plant", "query": "What object should I use to water a plant?", "description": "Water plant"},
    {"id": 5, "name": "get_lemon_out_of_tea", "query": "What object should I use to get lemon out of tea?", "description": "Get lemon out of tea"},
    {"id": 6, "name": "dig_hole", "query": "What object should I use to dig a hole?", "description": "Dig hole"},
    {"id": 7, "name": "open_bottle_of_beer", "query": "What object should I use to open a bottle of beer?", "description": "Open bottle of beer"},
    {"id": 8, "name": "open_parcel", "query": "What object should I use to open a parcel?", "description": "Open parcel"},
    {"id": 9, "name": "serve_wine", "query": "What object should I use to serve wine?", "description": "Serve wine"},
    {"id": 10, "name": "pour_sugar", "query": "What object should I use to pour sugar?", "description": "Pour sugar"},
    {"id": 11, "name": "smear_butter", "query": "What object should I use to smear butter?", "description": "Smear butter"},
    {"id": 12, "name": "extinguish_fire", "query": "What object should I use to extinguish a fire?", "description": "Extinguish fire"},
    {"id": 13, "name": "pound_carpet", "query": "What object should I use to pound a carpet?", "description": "Pound carpet"},
]


class Stage2AQueryInterface:
    """Stage 2A Query Interface for answering the 14 COCO-Tasks questions."""

    def __init__(self, device: str = "cpu"):
        """Initialize the query interface with the task-aware object selector."""
        print("Initializing Stage 2A Query Interface...")
        self.selector = TaskAwareObjectSelector(device=device)
        self.device = device
        print("Interface ready.\n")

    def get_task_by_query(self, query_text: str) -> dict:
        """Find the matching task based on query text. Supports partial matching."""
        query_lower = query_text.lower()

        phrase_order = [
            (0, ["step on", "step on something"]),
            (1, ["sit comfortably", "sit"]),
            (2, ["place flowers", "flowers"]),
            (3, ["potatoes", "out of fire"]),
            (4, ["water plant", "water a plant", "water"]),
            (5, ["lemon", "tea"]),
            (6, ["dig hole", "dig", "hole"]),
            (7, ["open bottle of beer", "bottle of beer", "beer"]),
            (8, ["open parcel", "parcel"]),
            (9, ["serve wine", "wine"]),
            (10, ["pour sugar", "sugar"]),
            (11, ["smear butter", "butter"]),
            (12, ["extinguish fire", "extinguish"]),
            (13, ["pound carpet", "carpet", "pound"]),
        ]

        for task_id, keywords in phrase_order:
            for keyword in keywords:
                if keyword in query_lower:
                    return COCO_TASKS[task_id]

        for task in COCO_TASKS:
            if task["description"].lower() in query_lower:
                return task

        print(f"Warning: No exact match for query '{query_text}', defaulting to task 1.")
        return COCO_TASKS[0]

    def answer_query(self, image_path: str, query_text: str, visualize: bool = True, output_path: str = None) -> dict:
        """Answer a single query for the given image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        task = self.get_task_by_query(query_text)
        task_id = task["id"]
        result = self.selector.select_object(image, task_id, apply_compatibility_filter=False)

        answer = {
            "query": task["query"],
            "task_id": task_id,
            "task_name": task["name"],
            "task_description": task["description"],
            "detected_objects": result["best_index"] is not None,
        }

        if result["best_index"] is not None:
            class_name = COCO_CLASSES.get(result["class_id"], f"unknown_{result['class_id']}")
            answer["selected_object"] = {
                "class": class_name,
                "class_id": result["class_id"],
                "confidence_score": result["best_score"],
                "bounding_box": result["box"],
            }
            answer["natural_language_answer"] = (
                f"For '{task['description']}', the most appropriate object is: "
                f"{class_name} (confidence: {result['best_score']:.2%})"
            )
        else:
            answer["selected_object"] = None
            answer["natural_language_answer"] = (
                f"No suitable object found for '{task['description']}' in the image."
            )

        if visualize and result["best_index"] is not None:
            vis_image = self._create_visualization(image, result, task, class_name=class_name)
            if output_path:
                cv2.imwrite(output_path, vis_image)
            answer["visualization_path"] = output_path
            answer["visualization"] = vis_image

        return answer

    def answer_all_queries(self, image_path: str, output_dir: str = "stage2a_results", visualize: bool = True) -> list:
        """Answer all 14 queries for the given image."""
        os.makedirs(output_dir, exist_ok=True)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        answers = []
        for task in COCO_TASKS:
            task_id = task["id"]
            result = self.selector.select_object(image, task_id, apply_compatibility_filter=False)

            answer = {
                "query": task["query"],
                "task_id": task_id,
                "task_name": task["name"],
                "task_description": task["description"],
                "detected_objects": result["best_index"] is not None,
            }

            if result["best_index"] is not None:
                class_name = COCO_CLASSES.get(result["class_id"], f"unknown_{result['class_id']}")
                answer["selected_object"] = {
                    "class": class_name,
                    "class_id": result["class_id"],
                    "confidence_score": result["best_score"],
                    "bounding_box": result["box"],
                }
                answer["natural_language_answer"] = (
                    f"For '{task['description']}', the most appropriate object is: "
                    f"{class_name} (confidence: {result['best_score']:.2%})"
                )
            else:
                answer["selected_object"] = None
                answer["natural_language_answer"] = (
                    f"No suitable object found for '{task['description']}' in the image."
                )

            if visualize and result["best_index"] is not None:
                vis_path = os.path.join(output_dir, f"task_{task_id:02d}_{task['name']}.png")
                vis_image = self._create_visualization(image, result, task, class_name=class_name)
                cv2.imwrite(vis_path, vis_image)
                answer["visualization_path"] = vis_path

            answers.append(answer)

        json_path = os.path.join(output_dir, "all_answers.json")
        with open(json_path, "w") as f:
            json.dump(answers, f, indent=2)

        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("Stage 2A - All 14 Task Answers Summary\n")
            f.write("=" * 50 + "\n\n")
            for ans in answers:
                f.write(f"Task {ans['task_id']+1}: {ans['task_description']}\n")
                f.write(f"  Query: {ans['query']}\n")
                f.write(f"  Answer: {ans['natural_language_answer']}\n\n")

        return answers

    def _create_visualization(self, image: np.ndarray, result: dict, task: dict, class_name: str = None) -> np.ndarray:
        """Create a visualization of the detected object."""
        vis = image.copy()
        h, w = vis.shape[:2]

        box = result["box"]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{task['description']}: {class_name}"
        score = f"{result['best_score']:.2%}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_w, label_h = label_size

        cv2.rectangle(vis, (x1, y1 - label_h - baseline - 10), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1, y1 - baseline - 5), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(vis, score, (x1, y1 - label_h - 15), font, font_scale * 0.8, (0, 0, 0), 1)

        return vis


def main():
    parser = argparse.ArgumentParser(description="Stage 2A Query Interface - Answer 14 COCO-Tasks questions")
    parser.add_argument("--image", "-i", type=str, default="data/coco_tasks/sample.jpg", help="Path to input image")
    parser.add_argument("--query", "-q", type=str, default=None, help="Natural language query (e.g., 'cut something')")
    parser.add_argument("--task-id", "-t", type=int, default=None, help="Task ID (0-13) for direct task selection")
    parser.add_argument("--all", "-a", action="store_true", help="Answer all 14 queries")
    parser.add_argument("--output-dir", "-o", type=str, default="stage2a_results", help="Output directory for results")
    parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu"], help="Device to use for inference (CPU only for Stage 2A)")
    args = parser.parse_args()

    interface = Stage2AQueryInterface(device=args.device)

    if args.all:
        print(f"Answering all 14 queries for image: {args.image}\n")
        answers = interface.answer_all_queries(args.image, output_dir=args.output_dir, visualize=True)
        print(f"\nResults saved to: {args.output_dir}/")
        print(f"  - all_answers.json (detailed results)")
        print(f"  - summary.txt (text summary)")
        print(f"  - task_XX_*.png (visualizations)\n")
        for ans in answers:
            print(ans["natural_language_answer"])
    elif args.query:
        print(f"Processing query: {args.query}\n")
        answer = interface.answer_query(args.image, args.query, visualize=True, output_path=os.path.join(args.output_dir, "query_result.png"))
        print(f"\nResult: {answer['natural_language_answer']}")
    elif args.task_id is not None:
        if not (0 <= args.task_id < 14):
            print("Error: task_id must be between 0 and 13")
            return
        task = COCO_TASKS[args.task_id]
        print(f"Processing task {args.task_id + 1}: {task['description']}\n")
        answer = interface.answer_query(args.image, task["query"], visualize=True, output_path=os.path.join(args.output_dir, f"task_{args.task_id:02d}.png"))
        print(f"\nResult: {answer['natural_language_answer']}")
    else:
        print("Stage 2A Query Interface")
        print("=" * 50)
        print("\nAvailable tasks (14 COCO-Tasks):")
        for task in COCO_TASKS:
            print(f"  {task['id'] + 1}. {task['query']}")
        print("\nUsage:")
        print("  --image <path> --query 'cut something'  # Single query")
        print("  --image <path> --task-id 0              # By task ID")
        print("  --image <path> --all                    # All 14 queries")


if __name__ == "__main__":
    main()