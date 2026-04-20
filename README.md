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

### 2. Install core dependencies

```bash
# Option A — Make (recommended)
make install

# Option B — pip directly
pip install -r requirements.txt
```

> **Apple Silicon (M-series) note:** `tensorflow-metal` is installed automatically on macOS ARM64 to enable GPU acceleration via Metal. It is skipped on Intel Macs and other platforms.

### 3. Install notebook dependencies (Jupyter only)

Only needed if you want to run the notebooks interactively:

```bash
# Option A — Make (recommended)
make install-notebooks

# Option B — pip directly
pip install -r notebooks/requirements.txt
```

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
