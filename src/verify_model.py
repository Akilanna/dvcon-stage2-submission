# Model Verification Script with Improved Ranking Algorithm
# Verifies the trained ranking model with custom images.
# Implements:
# 1. Task-gated scene descriptor (matches training configuration)
# 2. Task-object compatibility filter (semantic safety net)
# Usage: python verify_model.py --image path/to/image.jpg [--task task_id]
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import cv2
import numpy as np
import torch
from models.detector.yolo_wrapper import YOLODetector
from models.feature_module.feature_extractor import FeatureExtractor
from models.feature_module.scene_descriptor import SceneDescriptor
from models.task_encoder.task_embedding import TaskEmbeddingManager
from models.ranker.ranking_mlp import RankingMLP, build_ranking_input

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
    "pound carpet"
]

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
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

TASK_RELEVANT_CLASSES = {
    0: ["chair", "couch", "bed", "bench", "dining table"],  # step on something
    1: ["chair", "couch", "bed", "toilet", "bench", "dining table"],  # sit comfortably
    2: ["vase", "potted plant", "bottle", "cup", "bowl"],  # place flowers
    3: ["fork", "spoon", "knife"],  # get potatoes out of fire
    4: ["bottle", "cup", "bowl", "sink"],  # water plant
    5: ["spoon", "fork", "knife", "cup", "bowl"],  # get lemon out of tea
    6: ["spoon", "fork"],  # dig hole
    7: ["bottle", "knife", "spoon"],  # open bottle of beer
    8: ["scissors", "knife"],  # open parcel
    9: ["wine glass", "bottle", "cup"],  # serve wine
    10: ["bottle", "cup", "bowl", "spoon"],  # pour sugar
    11: ["knife", "spoon"],  # smear butter
    12: ["fire hydrant", "bottle", "cup", "bowl", "sink"],  # extinguish fire
    13: ["baseball bat", "tennis racket"]  # pound carpet
}

TASK_COMPATIBILITY_PENALTY = 0.01

class ModelVerifier:
    """Verify the trained ranking model with improved algorithm."""
    
    def __init__(self, yolo_model="yolov8n.pt", embedding_file="data/embeddings/tasks.npy",
                 ranker_ckpt="checkpoints/ranker_best.pt", projector_ckpt="checkpoints/projector.pt",
                 device="cpu", use_gating=True, det_conf_threshold=0.20):
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
            print(f"[!] WARNING: Projector not found at {projector_ckpt}")
        
        if os.path.isfile(ranker_ckpt):
            sd = torch.load(ranker_ckpt, map_location=device, weights_only=False)
            if isinstance(sd, dict) and "fc1.weight" in sd:
                self.ranker = RankingMLP(input_dim=731, hidden_dim=128).to(device)
                self.ranker.load_state_dict(sd)
            elif isinstance(sd, dict) and "model" in sd:
                self.ranker = RankingMLP(input_dim=731, hidden_dim=128).to(device)
                self.ranker.load_state_dict(sd["model"])
                if "scene_descriptor" in sd and use_gating:
                    self.scene_descriptor = SceneDescriptor(use_gating=True).to(device)
                    self.scene_descriptor.load_state_dict(sd["scene_descriptor"])
                    print(f"[+] Scene descriptor loaded with gating=True")
                else:
                    print(f"[!] Scene descriptor: using {'gated' if use_gating else 'mean-pool'} mode")
            else:
                raise RuntimeError(f"Unrecognised checkpoint format: {list(sd.keys())}")
            print(f"[+] Ranker loaded from {ranker_ckpt}")
        else:
            raise FileNotFoundError(f"Ranker checkpoint not found: {ranker_ckpt}")
        
        self.ranker.eval()
        self.scene_descriptor.eval()
    
    @torch.no_grad()
    def verify(self, image_path, task_id, score_threshold=0.3, apply_compatibility_filter=True):
        """Run verification with improved ranking algorithm."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        h, w = image.shape[:2]
        print(f"\nImage: {image_path} ({w}x{h})")
        print(f"Task: {TASK_NAMES[task_id]} (task_id={task_id})")
        print(f"Scene descriptor: {'gated pooling' if self.use_gating else 'mean pooling'}")
        print("-" * 50)
        
        det = self.detector.detect(image)
        boxes = det["boxes"]
        class_ids = det["class_ids"]
        scores = det["scores"]
        n = boxes.shape[0]
        print(f"Detected {n} objects:")
        
        if n == 0:
            print(" No objects detected.")
            return {"image": image, "best_index": None, "best_score": 0.0, "box": None,
                    "class_id": None, "all_boxes": [], "no_appropriate_object": True}
        
        print("\nObject analysis:")
        for i, (box, cls, conf) in enumerate(zip(boxes, class_ids, scores)):
            print(f" [{i}] {COCO_CLASSES[cls]} (det_conf={conf:.3f})")
        
        features = self.feature_extractor(image, boxes)
        task_embed = self.task_manager.get_embedding(task_id).to(features.device)
        scene = self.scene_descriptor.compute(features, task_embed if self.use_gating else None)
        ranking_input = build_ranking_input(features, scene, task_embed, class_ids)
        logits = self.ranker(ranking_input.to(self.device))
        ranking_scores = torch.sigmoid(logits)
        ranking_scores_np = ranking_scores.detach().cpu().numpy().copy()
        
        print(f"\nOriginal ranking scores: {ranking_scores_np.round(3)}")
        
        if apply_compatibility_filter:
            relevant_classes = TASK_RELEVANT_CLASSES.get(task_id, [])
            compatible_indices = [i for i, cls in enumerate(class_ids) if COCO_CLASSES[cls] in relevant_classes]
            if compatible_indices:
                print(f"\n[Compatibility Filter] Compatible objects: {compatible_indices}")
                incompatible_indices = [i for i in range(n) if i not in compatible_indices]
                if incompatible_indices:
                    print(f"[Compatibility Filter] Penalizing incompatible objects: {incompatible_indices}")
                    for idx in incompatible_indices:
                        ranking_scores_np[idx] *= TASK_COMPATIBILITY_PENALTY
                    print(f"Adjusted ranking scores: {ranking_scores_np.round(3)}")
            else:
                print(f"\n[Compatibility Filter] No objects match expected classes: {relevant_classes}")
                ranking_scores_np *= TASK_COMPATIBILITY_PENALTY
                print(f"Adjusted ranking scores: {ranking_scores_np.round(3)}")
        
        best_idx = int(ranking_scores_np.argmax())
        best_score = ranking_scores_np[best_idx]
        relevant_classes = TASK_RELEVANT_CLASSES.get(task_id, [])
        is_compatible = COCO_CLASSES[class_ids[best_idx]] in relevant_classes
        is_low_score = best_score < score_threshold
        no_appropriate = not is_compatible or is_low_score

        selected_class = COCO_CLASSES[class_ids[best_idx]]
        print(f"\nSelected object: {selected_class} (rank_score={best_score:.3f})")
        
        if no_appropriate:
            reason = "Incompatible class" if not is_compatible else "Low score"
            print(f"\n[!] Selection flagged: {reason}")
            print(f" Expected classes for '{TASK_NAMES[task_id]}': {relevant_classes}")
        
        return {
            "image": image, "best_index": best_idx, "best_score": best_score,
            "box": boxes[best_idx].tolist(), "class_id": class_ids[best_idx].item(),
            "all_boxes": list(zip(boxes, class_ids, scores, ranking_scores_np)),
            "no_appropriate_object": no_appropriate, "is_compatible": is_compatible,
            "task_id": task_id, "score_threshold": score_threshold
        }
    
    def visualize(self, result, output_path=None):
        """Visualize result using strict match-only bounding box drawing."""
        image = result["image"].copy()
        relevant_classes = TASK_RELEVANT_CLASSES.get(result.get('task_id', 0), [])
        
        if result["best_index"] is None:
            cv2.putText(image, "No objects detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            best_index = result["best_index"]
            best_box, best_cls, best_conf, best_rank_score = result["all_boxes"][best_index]
            is_match = (
                result.get("is_compatible", False)
                and best_rank_score >= float(result.get("score_threshold", 0.3))
                and COCO_CLASSES[best_cls] in relevant_classes
            )

            # Draw one box only when the selected object is a true task match.
            if is_match:
                x1, y1, x2, y2 = map(int, best_box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
                label = f"MATCH {COCO_CLASSES[best_cls]} det:{best_conf:.2f} rank:{best_rank_score:.2f}"
                cv2.putText(image, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            task_text = f"Task: {TASK_NAMES[result.get('task_id', 0)]}"
            cv2.putText(image, task_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            if result.get("no_appropriate_object", False):
                cv2.putText(image, "[NO COMPATIBLE OBJECT DETECTED - BLIND SPOT]", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(image, "[Compatible object found]", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if not is_match:
                cv2.putText(image, "[NO MATCH -> NO BOUNDING BOX DRAWN]", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"[+] Visualization saved to: {output_path}")
        return image

def main():
    parser = argparse.ArgumentParser(description="Verify trained ranking model (improved algorithm)")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--task", type=int, default=0, help="Task ID (0-13), default: 0 (cut)")
    parser.add_argument("--output", type=str, default=None, help="Output visualization path")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold (default: 0.3)")
    parser.add_argument("--det-conf", type=float, default=0.20, help="YOLO confidence threshold (default: 0.20)")
    parser.add_argument("--no-gating", action="store_true", help="Disable task-gated pooling")
    parser.add_argument("--no-filter", action="store_true", help="Disable compatibility filter")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    print("=" * 60)
    print(" Model Verification Script (Improved Algorithm)")
    print("=" * 60)
    print("\nAvailable tasks:")
    for i, name in enumerate(TASK_NAMES):
        print(f" {i}: {name}")
    
    verifier = ModelVerifier(
        device=args.device,
        use_gating=not args.no_gating,
        det_conf_threshold=args.det_conf,
    )
    args.task_id = args.task
    result = verifier.verify(args.image, args.task, args.threshold, apply_compatibility_filter=not args.no_filter)
    
    output_path = args.output or f"verification_result_{args.task}_{os.path.basename(args.image)}"
    if not output_path.endswith(".png"):
        output_path += ".png"
    
    verifier.visualize(result, output_path)
    print("\n" + "=" * 60)
    print(" Verification complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()