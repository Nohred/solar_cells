import numpy as np
import torch
import sys
from pathlib import Path
from elpv_dataset.utils import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.main import _encode_labels


def main():
	images, proba, types = load_dataset()
	labels = _encode_labels(proba, types, "binary", 0.5)
	indices = torch.randperm(
		len(labels), generator=torch.Generator().manual_seed(42)
	)
	train_size = int(len(labels) * 0.7)
	val_size = int(len(labels) * 0.2)
	test_indices = indices[train_size + val_size:].tolist()

	test_images = np.asarray([images[index] for index in test_indices])
	test_labels = labels[test_indices].numpy()
	output_dir = "data/test_split"
	import os
	os.makedirs(output_dir, exist_ok=True)
	np.save(f"{output_dir}/test_images.npy", test_images)
	np.save(f"{output_dir}/test_labels.npy", test_labels)
	np.save(f"{output_dir}/test_indices.npy", np.asarray(test_indices))
	print(f"Saved {len(test_labels)} test samples to {output_dir}")


if __name__ == "__main__":
	main()