# PathSight

A tissue classification model built with Convolutional Neural Networks (CNN).
The project trains a binary classifier on histopathology patch images to distinguish between tumour and normal tissue, using the [PatchCamelyon (PCam)](https://github.com/basveeling/pcam) dataset.

---

## Project structure

```
pathsight/
├── model/              # Model source code (preprocessing, config, etc.)
│   └── src/
├── notebooks/          # Exploratory and training notebooks
├── data/               # Raw and processed datasets (not tracked by git)
├── artifacts/          # Saved model weights and outputs
├── requirements.txt    # Core project dependencies
└── notebooks/requirements.txt  # Additional Jupyter-only dependencies
```

---

## Prerequisites

- Python **3.10.6** (managed via [pyenv](https://github.com/pyenv/pyenv), pinned in `.python-version`)
- `pip`

---

## Installation

### 1. Set up the pyenv environment

```bash
pyenv install 3.10.6          # skip if already installed
pyenv virtualenv 3.10.6 pathsight
pyenv activate pathsight      # or: pyenv shell pathsight
```

### 2. Install dependencies

| Command | What it does |
|---|---|
| `make install` | Core model deps only — clean-syncs and removes any stale packages |
| `make install-notebooks` | Core deps first, then adds Jupyter notebook deps |
| `make install-all` | Everything in one shot (core + notebooks) |

```bash
make install          # model/training work only
make install-all      # if you also want to run notebooks
```

> **Apple Silicon (M-series) note:** `tensorflow-metal` is installed automatically on macOS ARM64 to enable GPU acceleration via Metal. It is skipped on Intel Macs and other platforms.

> **Note:** All install commands use `pip-sync` under the hood, which removes any packages not listed in the requirements files — keeping the env clean and reproducible.

---
## Running the notebooks

```bash
jupyter notebook notebooks/
```

---

## Running tests

```bash
make test
```
