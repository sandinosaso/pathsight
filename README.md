# PathSight

PathSight is a cancer detection tool for histopathology images. It classifies
96×96 px tissue patches as **metastatic** or **normal** using transfer-learned
convolutional networks trained on the
[PatchCamelyon (PCam)](https://github.com/basveeling/pcam) dataset (~327 000
labelled images from the Camelyon16 sentinel-lymph-node challenge).

A user uploads a patch image → the backend runs inference → the frontend
displays the predicted label, confidence score, and full model metrics.

---
 
## Presentation
 
📊 [View Project Presentation](https://docs.google.com/presentation/d/1-_Y20zA1YZdxxyWPSzF0tJuOg4pkxX0_gHba0JrX7Ec/edit?usp=sharing)
 
---

## Contributors
 
| Avatar | Name | GitHub |
|--------|------|--------|
| <img src="https://avatars.githubusercontent.com/u/466105?v=4" width="50" height="50" style="border-radius:50%"> | **Sandino** | [@sandinosaso](https://github.com/sandinosaso) |
| <img src="https://avatars.githubusercontent.com/u/85241348?v=4" width="50" height="50" style="border-radius:50%"> | **Shayan Ghavami** | [@shayanghavami](https://github.com/shayanghavami) |
| <img src="https://avatars.githubusercontent.com/u/10108648?v=4" width="50" height="50" style="border-radius:50%"> | **Souheib Selmi** | [@sooheib](https://github.com/sooheib) |

---

## Repository Structure

```text
pathsight/
├── backend/               ← FastAPI inference API
│   ├── src/
│   │   ├── main.py        ← app entry point, /api/predict endpoint
│   │   ├── schemas.py     ← response shapes
│   │   └── logic/         ← model loading + post-processing
│   ├── scripts/           ← download_model.py (fetch from GCS)
│   └── Makefile           ← Cloud Run deployment
│
├── frontend/              ← React + TypeScript (Vite)
│   └── src/
│
├── model/                 ← training & evaluation library
│   ├── configs/
│   │   └── benchmarks.yaml    ← experiment matrix
│   ├── scripts/
│   │   └── run_benchmark.py   ← CLI trainer
│   └── src/model_service/     ← preprocessing, backbones, metrics
│
├── notebooks/             ← exploratory analysis and experiments
├── artifacts/             ← model weights and benchmark outputs (git-ignored)
├── data/                  ← PCam dataset cache (git-ignored)
│
├── Dockerfile             ← single image: backend + baked-in model
├── Makefile               ← top-level dev tasks
├── requirements.txt       ← all Python dependencies
└── .env.example           ← copy to .env and fill in your values
```

---

## Prerequisites

### Python (backend + model training)

Python **3.10.6** managed via [pyenv](https://github.com/pyenv/pyenv):

```bash
# Install pyenv (macOS)
brew install pyenv

# Install the pinned Python version and create a virtual env
pyenv install 3.10.6
pyenv virtualenv 3.10.6 pathsight
pyenv activate pathsight
```

The `.python-version` file in the repo root automatically activates the
`pathsight` env whenever you `cd` into the repo.

> **Apple Silicon (M-series):** `tensorflow-metal` is included in
> `requirements.txt` for macOS ARM64, enabling GPU acceleration via Metal.
> ConvNeXt backbones are run on CPU (Metal XLA incompatibility) — this is
> handled automatically.

### Node.js (frontend)

Node **18+** is required. Recommended install via [nvm](https://github.com/nvm-sh/nvm):

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
# Restart terminal, then:
nvm install 20 && nvm use 20
```

Or via Homebrew: `brew install node`

---

## Environment Configuration

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

Key variables for local development:

```env
APP_PORT=8000                            # Backend port
BEST_MODEL_PATH=artifacts/models/best_model.keras  # Local model path
DOCKER_IMAGE_NAME=pathsight-backend
DOCKER_LOCAL_PORT=8080
MODEL_BUCKET_NAME=your-gcs-bucket-name  # GCS bucket with best_model.keras
```

---

## Running Locally (without Docker)

### 1. Install all dependencies

```bash
make install-all
# Installs: core model library + notebooks + backend + model package (editable)
```

### 2. Download the model

You need a trained `.keras` model and its `.json` sidecar in
`artifacts/models/` before starting the backend.

**Option A — download from GCS** (requires `MODEL_BUCKET_NAME` in `.env`):

```bash
python backend/scripts/download_model.py
```

**Option B — use a locally trained model** (see [Training a model](#training-a-model)):

```bash
# After a benchmark run completes, copy its outputs to the models folder
# Files are named after the run_id (e.g. efficientnetb0_128_best.keras)
cp artifacts/benchmarks/<run_id>/<run_id>_best.keras artifacts/models/best_model.keras
cp artifacts/benchmarks/<run_id>/<run_id>_best.json  artifacts/models/best_model.json
```

### 3. Start the backend

```bash
make run-api-dev
# FastAPI with hot-reload starts on APP_PORT (default: 8000)
# Swagger docs: http://localhost:8000/docs
```

### 4. Start the frontend

In a second terminal:

```bash
make run-frontend-dev
# Vite dev server starts on http://localhost:5173
# /api requests are proxied to http://localhost:8000
```

Open **http://localhost:5173** in your browser.

---

## Training a Model

See [`backend/README.md`](backend/README.md) for the full training and
deployment guide. Quick start:

```bash
# Run a single experiment (downloads PCam on first run, ~7 GB)
python model/scripts/run_benchmark.py \
  --config model/configs/benchmarks.yaml \
  --only resnet50_96

# When it finishes, promote the result to the active model
cp artifacts/benchmarks/resnet50_96/resnet50_96_best.keras artifacts/models/best_model.keras
cp artifacts/benchmarks/resnet50_96/resnet50_96_best.json  artifacts/models/best_model.json
```

---

## Running Tests

```bash
make test
```

---

## Docker

```bash
# Build and run locally (requires model files in artifacts/models/)
make docker-up

# Deploy to Google Cloud Run (see backend/README.md for full instructions)
GCP_PROJECT_ID=your-project make -C backend docker_up
```

---

## More Documentation

| Topic | Where to look |
|---|---|
| Benchmark runner, YAML config, training artifacts, GCS upload | [`backend/README.md`](backend/README.md) |
| Frontend dev setup, Vite proxy, build commands | [`frontend/README.md`](frontend/README.md) |
| API endpoints and response schema | [`backend/README.md`](backend/README.md) → API Endpoints |
| Environment variables reference | [`.env.example`](.env.example) |
