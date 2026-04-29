# Backend — Cancer Detection API

FastAPI inference service. Loads a trained Keras model at startup and exposes
a `/api/predict` endpoint. Deployed to Google Cloud Run.

For general project setup (Python env, Node, `.env` config, running locally)
see the [root README](../README.md).

---

## Tech Stack

- **Python 3.10.6**, FastAPI, Uvicorn
- **TensorFlow 2.16** — CPU-only on Linux / Cloud Run
- **Docker** (root `Dockerfile`, build context = repo root)
- **Google Cloud Run** via `backend/Makefile`

---

## Project Structure

```text
backend/
├── Makefile                       ← Cloud Run deployment (docker_up)
├── scripts/
│   └── download_model.py          ← fetch best_model.keras + .json from GCS
└── src/
    ├── main.py                    ← FastAPI app, lifespan model load, /predict
    ├── schemas.py                 ← PredictionResponse / PredictionMeta
    ├── examples/                  ← bundled sample images for /api/examples
    └── logic/
        ├── predict.py             ← LoadedModel dataclass + load_model_trained()
        └── postprocessprediction.py
```

---

## API Endpoints

Once running, Swagger UI is at `http://localhost:<APP_PORT>/docs`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check — `{"status": "ok"}` |
| `GET` | `/api/examples` | List bundled example images |
| `GET` | `/api/examples/{id}/image` | Serve an example image as PNG |
| `POST` | `/api/predict` | Upload an image, receive cancer probability + model summary |

### POST /api/predict — response shape

```json
{
  "predicted_label": "cancer",
  "confidence": 0.87,
  "probabilities": {
    "cancer": 0.87,
    "no-cancer": 0.13
  },
  "original_base64": "<png-as-base64>",
  "heatmap_base64": null,
  "overlay_base64": null,
  "meta": {
    "input_size": [96, 96],
    "model_name": "resnet50 (best_model.keras)",
    "gradcam_layer": null
  },
  "model_summary": {
    "run_id": "resnet50_96",
    "backbone": "resnet50",
    "image_size": 96,
    "test": {
      "accuracy": 0.87, "recall": 0.91, "precision": 0.84,
      "f1": 0.87, "roc_auc": 0.95, "pr_auc": 0.93,
      "specificity": 0.84, "fnr": 0.09
    },
    "test_threshold": 0.2354,
    "thresholds": {
      "best_f1": 0.2354,
      "high_recall_95": 0.08,
      "high_precision_95": 0.74
    },
    "timing": { "inference_ms_per_image": 50 },
    "config": { "backbone": "resnet50", "image_size": 96, "...": "..." }
  }
}
```

`model_summary` is the full `<run_id>_best.json` sidecar loaded at startup. Use it in
the frontend to display training metrics, the decision threshold, and timing
info alongside a prediction.

---

## How the Model Is Loaded

The backend uses a FastAPI `lifespan` hook to load the model **after** Uvicorn
binds the port. This lets Cloud Run's startup probe succeed before the slow
Keras load begins.

`load_model_trained()` in `backend/src/logic/predict.py`:

1. Reads `BEST_MODEL_PATH` from the environment (default `artifacts/models/best_model.keras`)
2. Finds the JSON sidecar at the same path with a `.json` extension
3. Parses `backbone` and `image_size` from the sidecar
4. Derives `preprocess_mode` from the backbone name
5. Returns a `LoadedModel` dataclass containing the Keras model + all metadata

The sidecar file is the `<run_id>_best.json` produced by the benchmark runner (see below).

---

## Docker (local)

```bash
# Build image (root Dockerfile, context = repo root)
make docker-build-local

# Run built image
make docker-run-local

# Build + run in one command
make docker-up
```

The Dockerfile copies `artifacts/models/best_model.keras` and
`best_model.json` into the image at build time. Those files must exist locally
before building — download them with `download_model.py` or copy from a
completed benchmark run.

---

## Deploy to Cloud Run

```bash
# Build, push to Artifact Registry, and deploy — all in one command
GCP_PROJECT_ID=your-project-id make -C backend docker_up

# Override region (default: europe-central2)
GCP_PROJECT_ID=your-project-id GCP_REGION=us-central1 make -C backend docker_up
```

The deploy provisions Cloud Run with 4 GB RAM and 2 vCPUs. The GitHub Actions
workflow (`.github/workflows/deploy.yml`) calls this automatically on every
push to `main`, after first pulling the model files from GCS.

---

## Training & Benchmark Pipeline

### Running the benchmark script

```bash
# Run all active experiments in benchmarks.yaml
python model/scripts/run_benchmark.py --config model/configs/benchmarks.yaml

# Run a single experiment by run_id
python model/scripts/run_benchmark.py --config model/configs/benchmarks.yaml --only resnet50_96

# Quick smoke-test with 4 000 samples (fast — good for checking the pipeline)
python model/scripts/run_benchmark.py --config model/configs/benchmarks.yaml --max-samples 4000

# Full PCam dataset, no limit (best final metrics — slow)
python model/scripts/run_benchmark.py --config model/configs/benchmarks.yaml --max-samples 0

# Dry-run: print configs without training
python model/scripts/run_benchmark.py --config model/configs/benchmarks.yaml --dry-run

# Override the TFDS data directory (default: ~/tensorflow_datasets, ~7 GB)
python model/scripts/run_benchmark.py --config model/configs/benchmarks.yaml --data-dir /path/to/data
```

A sweep report (`model/configs/run_benchmark_report.json`) is written after
all runs with per-run status, elapsed time, AUC, and recall.

### Configuring benchmarks.yaml

Each list entry maps to a `RunConfig` field. Prefix a block with `#` to
disable it. Supported fields:

```yaml
- run_id: resnet50_96          # Unique ID — scopes all artifacts to artifacts/benchmarks/resnet50_96/
  backbone: resnet50           # resnet50 | convnexttiny | efficientnetb0 | mobilenetv3small | mobilenetv3large
  image_size: 96               # Input size in pixels (square). Common: 96, 128, 160, 224
  stage1_epochs: 10            # Epochs for head-only training (backbone frozen)
  stage2_epochs: 5             # Epochs for fine-tuning (top backbone layers unfrozen)
  fine_tune_layers: 30         # Number of backbone layers to unfreeze in Stage 2
  batch_size: 32               # Reduce to 16 for large images or low RAM
  head_units: 128              # Dense units in the classification head
  head_dropout: 0.3            # Dropout rate in the head
  learning_rate: 0.001         # Stage 1 learning rate
  fine_tune_lr: 0.00001        # Stage 2 fine-tuning learning rate
  max_train_samples: 20000     # Stage 1 cap (balanced 50/50); null = full ~262k dataset
  stage2_train_samples: 50000  # Stage 2 cap; null = reuse Stage 1 dataset
```

**To add a new backbone:**

1. Add a new entry to `benchmarks.yaml`.
2. Register the backbone in `model/src/model_service/training/backbones.py`
   — add it to `BackboneName`, `preprocess_mode()`, and `build_transfer_model()`.
3. Run with `--only <run_id>` to test.

### Decision threshold

All test metrics in `summary.json` / `<run_id>_best.json` are computed at
`best_f1_threshold` — the threshold that maximises F1 on the test set, derived
from the precision-recall curve (typically 0.2–0.4). This is lower than the
naive 0.5 and gives substantially higher recall (fewer missed cancers). The
threshold used is stored in the `test_threshold` field.

---

## Training Artifacts

After a run, all outputs land in `artifacts/benchmarks/<run_id>/`:

| File | Description |
|---|---|
| `<run_id>_best.keras` | Best Stage 2 checkpoint (saved by `ModelCheckpoint`) |
| `<run_id>_stage1_best.keras` | Best Stage 1 checkpoint (head-only) |
| `stage1_history.json` / `stage1.csv` | Per-epoch metrics for Stage 1 |
| `stage2_history.json` / `stage2.csv` | Per-epoch metrics for Stage 2 |
| `test_predictions.npz` | Raw `y_true` / `y_prob` arrays for offline analysis |
| `summary.json` | Full run summary — metrics, thresholds, timing, config |
| `<run_id>_best.json` | **Identical to `summary.json`** — GCS sidecar uploaded with the model |
| `confusion_matrix.png` | Confusion matrix at `best_f1_threshold` |
| `roc.png` | ROC curve with AUC |
| `pr_curve.png` | Precision-Recall curve with the operating point marked |

---

## Uploading to GCS

Only two files are needed to serve the model from the backend:

| File | GCS path |
|---|---|
| `best.keras` | `gs://<bucket>/best_model.keras` |
| `best.json` | `gs://<bucket>/best_model.json` |

**Upload via `make upload-model` (recommended):**

```bash
make upload-model MODEL_PATH=artifacts/benchmarks/resnet50_96/resnet50_96_best.keras
# The matching .json is detected automatically from the same path stem.
```

**Upload via raw `gcloud` CLI:**

```bash
BUCKET=pathsight-models-wagon-bootcamp-489111
RUN_ID=resnet50_96

gcloud storage cp artifacts/benchmarks/${RUN_ID}/${RUN_ID}_best.keras gs://${BUCKET}/best_model.keras
gcloud storage cp artifacts/benchmarks/${RUN_ID}/${RUN_ID}_best.json  gs://${BUCKET}/best_model.json
```

**Upload via Google Cloud Console:**

1. Go to [console.cloud.google.com/storage](https://console.cloud.google.com/storage)
2. Open your bucket (`pathsight-models-wagon-bootcamp-489111`)
3. Upload `<run_id>_best.keras` → rename to `best_model.keras`
4. Upload `<run_id>_best.json` → rename to `best_model.json`

The next CI/CD run (`deploy.yml`) will pull the new files and bake them into
the Docker image automatically.
