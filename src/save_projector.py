"""
Save FeatureExtractor projector weights to checkpoints/projector.pt

The projector (nn.Linear 512→128) now uses a fixed seed=42 for initialization,
so it is deterministic. Run this once to create the checkpoint file that
inference/run_pipeline.py loads at startup.

Usage:
    py scripts/save_projector.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models.feature_module.feature_extractor import FeatureExtractor

def main():
    os.makedirs("checkpoints", exist_ok=True)
    extractor = FeatureExtractor(feature_dim=128, device="cpu")
    out_path = "checkpoints/projector.pt"
    torch.save(extractor.projector.state_dict(), out_path)
    print(f"Projector state saved to {out_path}")
    w = extractor.projector.weight
    print(f"  weight shape: {w.shape}  mean={w.mean():.6f}  std={w.std():.6f}")

if __name__ == "__main__":
    main()
