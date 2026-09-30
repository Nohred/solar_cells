import os
from threading import Lock

from prometheus_client import Counter, Gauge, Histogram, start_http_server


_start_lock = Lock()
_started = False

INFERENCE_REQUESTS = Counter(
	"solar_cells_inference_requests_total",
	"Number of inference requests received by the GUI.",
	["status"],
)
IMAGES_PROCESSED = Counter(
	"solar_cells_images_processed_total",
	"Number of images sent to Triton.",
)
INFERENCE_LATENCY = Histogram(
	"solar_cells_inference_latency_seconds",
	"End-to-end Triton inference latency.",
	buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
ACTIVE_MODEL = Gauge(
	"solar_cells_active_model",
	"Whether the named model is the only READY model.",
	["model"],
)


def start_metrics_server() -> None:
	global _started
	with _start_lock:
		if not _started:
			if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
				_started = True
				return
			start_http_server(int(os.getenv("METRICS_PORT", "8502")))
			_started = True


def set_active_model(model_name: str | None) -> None:
	for allowed_name in ("resnet18", "efficientnet_b0"):
		ACTIVE_MODEL.labels(allowed_name).set(
			1 if model_name == allowed_name else 0
		)
