import torch
import numpy as np
import hashlib
import json
from pathlib import Path
from elpv_dataset.utils import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TEST_SPLIT_DIR = Path("data/test_split")
TEST_IMAGES_PATH = TEST_SPLIT_DIR / "test_images.npy"
TEST_LABELS_PATH = TEST_SPLIT_DIR / "test_labels.npy"
TEST_INDICES_PATH = TEST_SPLIT_DIR / "test_indices.npy"
SPLIT_MANIFEST_PATH = TEST_SPLIT_DIR / "split_manifest.json"


class ELPVDataset(Dataset):
	def __init__(self, images, labels, train=False):
		self.images = images
		self.labels = torch.as_tensor(labels, dtype=torch.long)
		augmentation = [
			transforms.RandomRotation(degrees=10),
			transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
		] if train else []
		self.transform = transforms.Compose([
			transforms.Resize((224, 224), antialias=True),
			*augmentation,
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
		])

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		image = torch.tensor(np.array(self.images[index], copy=True), dtype=torch.float32)
		if image.ndim == 2:
			image = image.unsqueeze(0)
		elif image.ndim == 3 and image.shape[-1] in (1, 3):
			image = image.permute(2, 0, 1)
		if image.ndim != 3:
			raise ValueError(f"Expected an image with 2 or 3 dimensions, got {tuple(image.shape)}")
		if image.max() > 1:
			image = image / 255.0
		if image.shape[0] == 1:
			image = image.repeat(3, 1, 1)
		elif image.shape[0] != 3:
			raise ValueError(f"Expected 1 or 3 channels, got {image.shape[0]}")
		return self.transform(image), self.labels[index]


def _encode_labels(proba, types, label_mode, defect_threshold):
	if label_mode == "binary":
		probabilities = torch.tensor(np.array(proba, copy=True), dtype=torch.float32)
		return (probabilities >= defect_threshold).long()
	if label_mode == "type":
		type_names = sorted(set(types))
		type_to_index = {name: index for index, name in enumerate(type_names)}
		return torch.tensor([type_to_index[name] for name in types], dtype=torch.long)
	raise ValueError("label_mode must be 'binary' or 'type'")


def _load_fixed_test_split():
	if not TEST_IMAGES_PATH.is_file() or not TEST_LABELS_PATH.is_file():
		return None
	test_images = np.load(TEST_IMAGES_PATH, allow_pickle=False)
	test_labels = torch.from_numpy(np.load(TEST_LABELS_PATH, allow_pickle=False)).long()
	test_indices = None
	if TEST_INDICES_PATH.is_file():
		test_indices = np.load(TEST_INDICES_PATH, allow_pickle=False).tolist()
	if len(test_images) != len(test_labels) or (
		test_indices is not None and len(test_images) != len(test_indices)
	):
		raise ValueError("Fixed test images and labels must have the same length")
	return test_images, test_labels, test_indices


def _dataset_signature(images, labels):
	hasher = hashlib.sha256()
	hasher.update(np.asarray(images).tobytes())
	hasher.update(np.asarray(labels).tobytes())
	return hasher.hexdigest()


def _stratified_partition(indices, labels, train_size, val_size, seed):
	rng = np.random.default_rng(seed)
	train_indices, val_indices = [], []
	for class_id in sorted(torch.unique(labels).tolist()):
		class_indices = np.asarray([index for index in indices if int(labels[index]) == class_id])
		rng.shuffle(class_indices)
		class_train = round(len(class_indices) * train_size / (train_size + val_size))
		train_indices.extend(class_indices[:class_train].tolist())
		val_indices.extend(class_indices[class_train:].tolist())
	rng.shuffle(train_indices)
	rng.shuffle(val_indices)
	return train_indices, val_indices


def _validate_split(manifest, labels, dataset_size, signature):
	if manifest.get("dataset_signature") != signature:
		raise ValueError("Saved split belongs to a different ELPV dataset or label configuration")
	all_indices = manifest["train_indices"] + manifest["val_indices"] + manifest["test_indices"]
	if len(all_indices) != dataset_size or len(set(all_indices)) != dataset_size:
		raise ValueError("Saved split indices are incomplete or overlap")
	if any(index < 0 or index >= dataset_size for index in all_indices):
		raise ValueError("Saved split contains an out-of-range index")
	if any(int(labels[index]) != int(label) for index, label in zip(manifest["test_indices"], manifest["test_labels"])):
		raise ValueError("Saved split labels do not match the current dataset")


def get_elpv_loaders(
	batch_size=64,
	val_split=0.2,
	test_split=0.1,
	label_mode="binary",
	defect_threshold=0.5,
	generator_seed=42,
):
	images, proba, types = load_dataset()
	labels = _encode_labels(proba, types, label_mode, defect_threshold)
	if val_split < 0 or test_split < 0 or val_split + test_split >= 1:
		raise ValueError("val_split and test_split must be non-negative and sum to less than 1")
	train_size = int(len(labels) * (1 - val_split - test_split))
	val_size = int(len(labels) * val_split)
	signature = _dataset_signature(images, labels.numpy())
	fixed_test_split = _load_fixed_test_split()
	if fixed_test_split is not None:
		test_images, test_labels, test_indices = fixed_test_split
		if test_indices is None:
			test_indices = []
			for test_image in test_images:
				matches = [
					index for index, image in enumerate(images)
					if np.array_equal(image, test_image)
				]
				if len(matches) != 1:
					raise ValueError("Fixed test split does not match the current ELPV dataset")
				test_indices.append(matches[0])
		if any(index < 0 or index >= len(labels) for index in test_indices):
			raise ValueError("Fixed test split does not match the current ELPV dataset")
		if any(int(labels[index]) != int(test_labels[position]) for position, index in enumerate(test_indices)):
			raise ValueError("Fixed test labels do not match the current ELPV labels")
		for position, index in enumerate(test_indices):
			if not np.array_equal(images[index], test_images[position]):
				raise ValueError("Fixed test split image IDs do not match the current ELPV dataset")
		fixed_test_indices = set(test_indices)
		available_indices = [index for index in range(len(labels)) if index not in fixed_test_indices]
		if SPLIT_MANIFEST_PATH.is_file():
			manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
			_validate_split(manifest, labels, len(labels), signature)
			train_indices = manifest["train_indices"]
			val_indices = manifest["val_indices"]
			test_indices = manifest["test_indices"]
		else:
			train_indices, val_indices = _stratified_partition(
				available_indices, labels, train_size, val_size, generator_seed
			)
			manifest = {
				"dataset_signature": signature,
				"seed": generator_seed,
				"defect_threshold": defect_threshold,
				"train_indices": train_indices,
				"val_indices": val_indices,
				"test_indices": test_indices,
				"test_labels": [int(labels[index]) for index in test_indices],
			}
			TEST_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
			SPLIT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
		train_dataset = ELPVDataset(images, labels, train=True)
		val_dataset = ELPVDataset(images, labels, train=False)
		test_dataset = ELPVDataset(test_images, test_labels, train=False)
		test_indices = list(range(len(test_labels)))
	if fixed_test_split is None:
		if SPLIT_MANIFEST_PATH.is_file():
			manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
			_validate_split(manifest, labels, len(labels), signature)
			train_indices = manifest["train_indices"]
			val_indices = manifest["val_indices"]
			test_indices = manifest["test_indices"]
		else:
			train_indices, val_indices = _stratified_partition(
				list(range(len(labels))), labels, train_size, val_size, generator_seed
			)
			test_indices = [index for index in range(len(labels)) if index not in set(train_indices + val_indices)]
			manifest = {
				"dataset_signature": signature,
				"seed": generator_seed,
				"defect_threshold": defect_threshold,
				"train_indices": train_indices,
				"val_indices": val_indices,
				"test_indices": test_indices,
				"test_labels": [int(labels[index]) for index in test_indices],
			}
			SPLIT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
		train_dataset = ELPVDataset(images, labels, train=True)
		val_dataset = ELPVDataset(images, labels, train=False)
		test_dataset = val_dataset

	print(
		f"Train size: {len(train_indices)}, Validation size: {len(val_indices)}, "
		f"Test size: {len(test_indices)}, "
		f"Classes: {int(labels.max()) + 1}"
	)
	return (
		DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True,
			generator=torch.Generator().manual_seed(generator_seed)),
		DataLoader(Subset(val_dataset, val_indices), batch_size=batch_size, shuffle=False),
		DataLoader(Subset(test_dataset, test_indices), batch_size=batch_size, shuffle=False),
	)

