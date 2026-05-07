"""
Generate task embeddings for COCO-Tasks.

This script loads the official 14 task descriptions from the COCO-Tasks
annotation-tool fixture and encodes them with MiniLM (384-dim).

Output: data/embeddings/tasks.npy  (shape: [14, 384])

Usage:
  py scripts/generate_task_embeddings.py
"""

import json
import os
import urllib.request

import numpy as np
from sentence_transformers import SentenceTransformer


FIXTURE_URL = (
	"https://raw.githubusercontent.com/coco-tasks/annotation-tool/"
	"master/src/cocoannot/annotpreferred/fixtures/tasks.json"
)


def load_task_descriptions() -> list:
	"""Load and return 14 task descriptions sorted by task number (1..14)."""
	with urllib.request.urlopen(FIXTURE_URL, timeout=60) as resp:
		tasks = json.loads(resp.read().decode("utf-8"))

	by_number = {}
	for entry in tasks:
		fields = entry["fields"]
		number = int(fields["number"])
		desc = fields["desc"].strip().replace("\r\n", " ")
		by_number[number] = desc

	if sorted(by_number.keys()) != list(range(1, 15)):
		raise RuntimeError("Expected task numbers 1..14 in fixture data")

	return [by_number[i] for i in range(1, 15)]


def main(
	model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
	output_file: str = "data/embeddings/tasks.npy",
):
	os.makedirs(os.path.dirname(output_file), exist_ok=True)

	task_descriptions = load_task_descriptions()
	print(f"Loaded {len(task_descriptions)} task descriptions")

	model = SentenceTransformer(model_name)
	embeddings = model.encode(
		task_descriptions,
		convert_to_numpy=True,
		normalize_embeddings=False,
		show_progress_bar=False,
	)

	if embeddings.shape != (14, 384):
		raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

	embeddings = embeddings.astype(np.float32)
	np.save(output_file, embeddings)

	print(f"Saved embeddings to: {output_file}")
	print(f"Shape: {embeddings.shape}")
	print(f"Dtype: {embeddings.dtype}")


if __name__ == "__main__":
	main()
