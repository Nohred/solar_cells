import os
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter

import numpy as np
import requests
from PIL import Image


MODEL_NAMES = ("resnet18", "efficientnet_b0")
CLASS_NAMES = ("no_defect", "defect")
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


class TritonClientError(RuntimeError):
	pass


@dataclass(frozen=True)
class Prediction:
	class_name: str
	confidence: float


class TritonClient:
	def __init__(self, base_url: str | None = None, timeout: float = 15.0):
		self.base_url = (base_url or os.getenv("TRITON_URL", "http://localhost:8000")).rstrip("/")
		self.timeout = timeout

	def _request(self, method: str, path: str, **kwargs) -> requests.Response:
		try:
			response = requests.request(
				method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
			)
			response.raise_for_status()
			return response
		except requests.RequestException as error:
			raise TritonClientError(f"Triton no está disponible: {error}") from error

	def ready_models(self) -> list[str]:
		response = self._request("POST", "/v2/repository/index")
		models = response.json()
		return [
			item["name"]
			for item in models
			if item.get("name") in MODEL_NAMES and item.get("state") == "READY"
		]

	def active_model(self) -> str:
		ready = self.ready_models()
		if len(ready) != 1:
			raise TritonClientError(
				"Se requiere exactamente un modelo READY en Triton "
				f"(encontrados: {', '.join(ready) or 'ninguno'})."
			)
		return ready[0]

	@staticmethod
	def _prepare_image(image_bytes: bytes) -> np.ndarray:
		image = Image.open(BytesIO(image_bytes)).convert("RGB")
		array = np.asarray(image.resize(IMAGE_SIZE), dtype=np.float32) / 255.0
		array = (array - IMAGENET_MEAN) / IMAGENET_STD
		return np.transpose(array, (2, 0, 1))

	def predict(self, images: list[bytes]) -> tuple[str, list[Prediction], float]:
		if not 1 <= len(images) <= 5:
			raise ValueError("El lote debe contener entre 1 y 5 imágenes.")
		model_name = self.active_model()
		batch = np.stack([self._prepare_image(image) for image in images]).astype(np.float32)
		body = {
			"inputs": [
				{
					"name": "input",
					"shape": list(batch.shape),
					"datatype": "FP32",
					"data": batch.reshape(-1).tolist(),
				}
			],
			"outputs": [{"name": "output"}],
		}
		started = perf_counter()
		response = self._request("POST", f"/v2/models/{model_name}/infer", json=body)
		latency = perf_counter() - started
		values = np.asarray(response.json()["outputs"][0]["data"], dtype=np.float32)
		scores = values.reshape(len(images), len(CLASS_NAMES))
		probabilities = self._softmax(scores)
		predictions = [
			Prediction(CLASS_NAMES[int(row.argmax())], float(row.max()))
			for row in probabilities
		]
		return model_name, predictions, latency

	@staticmethod
	def _softmax(values: np.ndarray) -> np.ndarray:
		shifted = values - values.max(axis=1, keepdims=True)
		exponentials = np.exp(shifted)
		return exponentials / exponentials.sum(axis=1, keepdims=True)
