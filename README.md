# Solar Cells Inspector

Deep learning pipeline for binary defect classification of solar cells using the ELPV dataset. The project fine-tunes two ImageNet-pretrained CNNs:

- ResNet18
- EfficientNet-B0

The trained models are exported to ONNX and served with NVIDIA Triton Inference Server. A Streamlit application allows users to upload up to five solar-cell images and view one prediction per image. Users do not select the model; the administrator controls the active model through Triton.

## Repository Overview

```text
artifacts/                  Saved model weights
callbacks/                  Training callbacks
datasets/                   ELPV dataset loading and preprocessing
engine/                     Training and evaluation loops
gui/                        Streamlit application and Triton client
models/                     ResNet18, EfficientNet-B0, and model factory
observability/              Prometheus and Grafana configuration
serving/                    ONNX export and Triton model control CLI
serving/model_repository/   Triton model repository
train.py                    Training entry point
```

The inference preprocessing matches the training pipeline: RGB images are resized to `224x224`, scaled to `[0, 1]`, and normalized with ImageNet mean and standard deviation.

## Requirements

- Docker and Docker Compose
- Conda with the `deeplearning` environment for local Python commands
- A trained checkpoint at `artifacts/best_model_<model>.pth` before exporting models

The Triton container can run on CPU. GPU support requires a compatible NVIDIA driver and NVIDIA Container Toolkit.

## Train a Model

Activate the project environment and run both reproducible two-phase trainings:

```bash
conda activate deeplearning
python train.py
```

Run one architecture or the head-only baseline with:

```bash
python train.py --model resnet18
python train.py --model efficientnet_b0
python train.py --model efficientnet_b0 --head-only
```

The script persists a stratified split manifest in `data/test_split/split_manifest.json`,
uses the same split for both models, selects checkpoints by validation macro F1, and
evaluates test once after selection. Class 1 is explicitly the defective class using
`defect_threshold=0.5`. Reports are saved as `artifacts/metrics_<model>.json`.

The first training run can materialize the deterministic test split so every model is evaluated on the same images:

```bash
conda activate deeplearning
python scripts/save_test_split.py
```

This creates the fixed test arrays and their source indices in `data/test_split/`. Once present, `get_elpv_loaders()` reuses that test set instead of generating it again.

Training saves separate phase and selected checkpoints to:

```text
artifacts/resnet18_head_only.pth
artifacts/resnet18_finetune.pth
artifacts/best_model_resnet18.pth
artifacts/best_model_efficientnet_b0.pth
```

The model architecture selected in `train.py` must match the architecture used during ONNX export.

## Export to ONNX

Select the architecture explicitly when exporting:

```bash
conda activate deeplearning
python serving/export_cnn_onnx.py --model efficientnet_b0
python serving/export_cnn_onnx.py --model resnet18
```

If ONNX import fails with a protobuf runtime-version error, update the environment
dependency before exporting, for example: `conda activate deeplearning && pip install -U protobuf`.

The exported file is written to the corresponding Triton directory:

```text
serving/model_repository/efficientnet_b0/1/model.onnx
serving/model_repository/resnet18/1/model.onnx
```

The exporter also validates the ONNX model against the PyTorch model.

## Start the Application

Start all services from the repository root:

```bash
docker compose up -d
```

The default active model is EfficientNet-B0. Triton runs in explicit model-control mode and loads only that model at startup.

Available services:

| Service | URL | Purpose |
| --- | --- | --- |
| Streamlit GUI | http://localhost:8501 | Upload images and view predictions |
| Prometheus | http://localhost:9090 | Metrics collection and queries |
| Grafana | http://localhost:3000 | Metrics dashboard |
| Triton HTTP API | http://localhost:8000 | Local inference and model-control API |

To access the GUI from another computer on the local network, use the host machine's IP address:

```text
http://HOST_IP:8501
```

The GUI accepts JPG, JPEG, PNG, and BMP files, with a maximum of five images per inference batch.

View service logs with:

```bash
docker compose logs -f gui triton
```

Stop the services with:

```bash
docker compose down
```

## Switch Models

Model administration is performed from the host with `serving/model_control.py`. The GUI has no model selector.

Check the current repository state:

```bash
conda activate deeplearning
python serving/model_control.py status
```

Switch the active model without restarting the application:

```bash
python serving/model_control.py switch resnet18
python serving/model_control.py switch efficientnet_b0
```

The `switch` command unloads the current model, loads the requested model, and verifies that exactly one allowed model is `READY`.

You can also load or unload a model explicitly:

```bash
python serving/model_control.py load resnet18
python serving/model_control.py unload resnet18
```

To choose the model loaded after a Triton restart, set `ACTIVE_MODEL` when starting the service:

```bash
ACTIVE_MODEL=resnet18 docker compose up -d --force-recreate triton
```

Valid values are `resnet18` and `efficientnet_b0`. The default is `efficientnet_b0`.

## Observability and Audit

Start the observability services together with the application:

```bash
docker compose up -d
```

### Grafana Dashboard

Open Grafana in a browser:

```text
http://localhost:3000
```

The default Grafana credentials are:

```text
Username: admin
Password: admin
```

Grafana normally asks you to change the password on the first login. The provisioned `Solar Cells` dashboard is available from **Dashboards** and shows inference request rate, p95 latency, processed images, and the active model.

From another computer on the local network, replace `localhost` with the host machine's IP address:

```text
http://HOST_IP:3000
```

### Prometheus

Open the Prometheus web interface at:

```text
http://localhost:9090
```

To verify that metrics are being collected, open **Status > Targets** or query the targets API:

```bash
curl http://localhost:9090/api/v1/targets
```

Both `solar-cells-gui` and `triton` should have the status `up`. Useful PromQL queries include:

```promql
sum by (status) (rate(solar_cells_inference_requests_total[5m]))
```

```promql
histogram_quantile(0.95, sum by (le) (rate(solar_cells_inference_latency_seconds_bucket[5m])))
```

```promql
nv_inference_request_success
```

The GUI exposes application metrics on the internal Docker network at `gui:8502`. Triton exposes its metrics at `triton:8002`. Triton's host ports are bound to `localhost` intentionally because its HTTP API also contains administrative model-control operations.

### Logs and Audit File

Follow observability logs with:

```bash
docker compose logs -f prometheus grafana
```

Inference audit events are stored as JSON Lines in:

```text
gui/data/audit.jsonl
```

The audit log stores operational metadata such as timestamp, batch size, model, latency, and status. Image contents are not stored.

## Troubleshooting

If the GUI reports that no model is `READY`, check the Triton state:

```bash
python serving/model_control.py status
```

Then load a model:

```bash
python serving/model_control.py switch efficientnet_b0
```

If the model fails to load, inspect Triton logs and verify that both the ONNX file and its `config.pbtxt` exist in the expected model directory.
