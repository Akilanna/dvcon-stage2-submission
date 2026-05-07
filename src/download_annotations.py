"""Download all 28 COCO-Tasks annotation files from GitHub LFS."""
import os
import urllib.request

BASE_URL = "https://media.githubusercontent.com/media/coco-tasks/dataset/cvpr2019/annotations/"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "annotations")
os.makedirs(OUT_DIR, exist_ok=True)

files = [f"task_{t}_{split}.json" for t in range(1, 15) for split in ("train", "test")]

for fname in files:
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path):
        print(f"SKIP (exists): {fname}")
        continue
    url = BASE_URL + fname
    print(f"Downloading {fname} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"OK ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"FAILED: {e}")

print("\nDone. Files in", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".json"):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1e6
        print(f"  {f}: {size:.1f} MB")
