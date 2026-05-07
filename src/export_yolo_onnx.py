"""
Export YOLO to ONNX Script
----------------------------
Exports the YOLOv8 model to ONNX format for edge deployment.

Responsibilities:
- Load a trained or pretrained YOLOv8 model
- Export to ONNX with specified input dimensions
- Validate the exported ONNX model
- Optionally quantize for INT8 inference

Output: models/detector/yolov8n.onnx

Usage: python scripts/export_yolo_onnx.py --model yolov8n --output models/detector/yolov8n.onnx
"""
