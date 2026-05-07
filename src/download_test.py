"""Download and inspect COCO-Tasks annotations from GitHub LFS."""
import urllib.request
import os
import json

BASE_URL = "https://media.githubusercontent.com/media/coco-tasks/dataset/cvpr2019/annotations"
OUT_DIR = "data/annotations/coco_tasks_raw"
os.makedirs(OUT_DIR, exist_ok=True)

# Test with task_1_train.json first
url = f"{BASE_URL}/task_1_train.json"
dest = os.path.join(OUT_DIR, "task_1_train.json")
print(f"Downloading {url} ...")
urllib.request.urlretrieve(url, dest)
size = os.path.getsize(dest)
print(f"Downloaded: {size} bytes")

with open(dest) as f:
    data = json.load(f)

print(f"Top-level keys: {list(data.keys())}")
if "annotations" in data:
    print(f"Annotation count: {len(data['annotations'])}")
    print(f"First record:\n{json.dumps(data['annotations'][0], indent=2)}")
if "images" in data:
    print(f"Image count: {len(data['images'])}")
    print(f"First image: {data['images'][0]}")
