#!/usr/bin/env python3
"""Inference pipeline for task-aware object selection.

Pipeline:
image + task_id -> YOLO -> feature extractor -> scene descriptor -> ranker
-> compatibility filter -> selected object
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.detector.yolo_wrapper import YOLODetector
from models.feature_module.feature_extractor import FeatureExtractor
from models.feature_module.scene_descriptor import SceneDescriptor
from models.ranker.ranking_mlp import RankingMLP, build_ranking_input
from models.task_encoder.task_embedding import TaskEmbeddingManager

RANKER_CHECKPOINT = "checkpoints/ranker_best.pt"
PROJECTOR_CHECKPOINT = "checkpoints/projector.pt"
TASK_COMPATIBILITY_PENALTY = 0.01

TASK_RELEVANT_CLASSES = {
    0: ["chair", "couch", "bed", "bench", "dining table"],
    1: ["chair", "couch", "bed", "toilet", "bench", "dining table"],
    2: ["vase", "potted plant", "bottle", "cup", "bowl"],
    3: ["fork", "spoon", "knife"],
    4: ["bottle", "cup", "bowl", "sink"],
    5: ["spoon", "fork", "knife", "cup", "bowl"],
    6: ["spoon", "fork"],
    7: ["bottle", "knife", "spoon"],
    8: ["scissors", "knife"],
    9: ["wine glass", "bottle", "cup"],
    10: ["bottle", "cup", "bowl", "spoon"],
    11: ["knife", "spoon"],
    12: ["fire hydrant", "bottle", "cup", "bowl", "sink"],
    13: ["baseball bat", "tennis racket"],
}

TASK_HINT_BOOST = 0.12

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class TaskAwareObjectSelector:
    """Task-aware object selection wrapper."""

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        embedding_file: str = "data/embeddings/tasks.npy",
        ranker_ckpt: str = RANKER_CHECKPOINT,
        projector_ckpt: str = PROJECTOR_CHECKPOINT,
        device: str = "cpu",
        use_gating: bool = True,
        det_conf_threshold: float = 0.20,
    ):
        self.device = device
        self.use_gating = use_gating

        self.detector = YOLODetector(
            model_name=yolo_model,
            device=device,
            conf_threshold=det_conf_threshold,
        )
        self.feature_extractor = FeatureExtractor(feature_dim=128, device=device)
        self.scene_descriptor = SceneDescriptor(use_gating=use_gating).to(device)
        self.task_manager = TaskEmbeddingManager(embedding_file)

        if os.path.isfile(projector_ckpt):
            proj_sd = torch.load(projector_ckpt, map_location=device, weights_only=True)
            self.feature_extractor.projector.load_state_dict(proj_sd)
            print(f"[+] Projector loaded from {projector_ckpt}")
        else:
            print(f"[!] WARNING: projector checkpoint not found at {projector_ckpt}")

        if not os.path.isfile(ranker_ckpt):
            raise FileNotFoundError(f"Ranker checkpoint not found: {ranker_ckpt}")

        sd = torch.load(ranker_ckpt, map_location=device, weights_only=False)
        self.ranker = RankingMLP(input_dim=731, hidden_dim=128).to(device)

        if isinstance(sd, dict) and "fc1.weight" in sd:
            self.ranker.load_state_dict(sd)
        elif isinstance(sd, dict) and "model" in sd:
            self.ranker.load_state_dict(sd["model"])
            if "scene_descriptor" in sd and use_gating:
                self.scene_descriptor.load_state_dict(sd["scene_descriptor"])
                print("[+] Scene descriptor loaded with gating=True")
        else:
            raise RuntimeError(f"Unrecognised checkpoint format: {list(sd.keys())}")

        print(f"[+] Ranker loaded from {ranker_ckpt}")
        self.ranker.eval()
        self.scene_descriptor.eval()

    @torch.no_grad()
    def select_object(
        self,
        image: np.ndarray,
        task_id: int,
        apply_compatibility_filter: bool = False,
        score_threshold: float = 0.3,
    ) -> dict:
        """Run full task-aware selection for one image."""
        det = self.detector.detect(image)
        boxes = det["boxes"]
        class_ids = det["class_ids"]
        n = boxes.shape[0]

        if n == 0:
            return {
                "best_index": None,
                "best_score": 0.0,
                "box": None,
                "class_id": None,
                "no_appropriate_object": True,
                "is_compatible": False,
                "all_scores": [],
            }

        features = self.feature_extractor(image, boxes)
        task_embed = self.task_manager.get_embedding(task_id).to(features.device)
        scene = self.scene_descriptor.compute(features, task_embed if self.use_gating else None)
        ranking_input = build_ranking_input(features, scene, task_embed, class_ids)

        logits = self.ranker(ranking_input.to(self.device))
        scores_np = torch.sigmoid(logits).detach().cpu().numpy().copy()

        if apply_compatibility_filter:
            relevant_classes = TASK_RELEVANT_CLASSES.get(task_id, [])
            compatible_indices = [
                i for i, cls in enumerate(class_ids)
                if COCO_CLASSES[int(cls)] in relevant_classes
            ]
            if compatible_indices:
                for idx in compatible_indices:
                    scores_np[idx] += TASK_HINT_BOOST
            else:
                # No matching classes: keep the raw ranking so general images still work.
                pass

        best_idx = int(scores_np.argmax())
        best_score = float(scores_np[best_idx])

        relevant_classes = TASK_RELEVANT_CLASSES.get(task_id, [])
        selected_class = COCO_CLASSES[int(class_ids[best_idx])]
        is_compatible = selected_class in relevant_classes
        is_low_score = best_score < score_threshold
        no_appropriate = is_low_score or (apply_compatibility_filter and (not is_compatible))

        return {
            "best_index": best_idx,
            "best_score": best_score,
            "box": boxes[best_idx].tolist(),
            "class_id": int(class_ids[best_idx].item()),
            "no_appropriate_object": no_appropriate,
            "is_compatible": is_compatible,
            "all_scores": scores_np.tolist(),
        }


if __name__ == "__main__":
    import cv2

    sample_path = "data/coco_tasks/sample.jpg"
    if os.path.isfile(sample_path):
        img = cv2.imread(sample_path)
    else:
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    selector = TaskAwareObjectSelector()
    result = selector.select_object(img, task_id=0)

    print("\n=== Result ===")
    for key, value in result.items():
        print(f" {key}: {value}")
