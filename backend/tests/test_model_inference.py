"""Smoke tests for model loading and inference.

The model's expected input size is read from the companion ``best_model.json``
sidecar (same directory, same stem) so these tests work correctly regardless of
which model is deployed — 96x96, 128x128, 224x224, etc.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_path() -> Path:
    return Path(os.getenv("BEST_MODEL_PATH", "artifacts/models/best_model.keras"))


def _sidecar_path(model_path: Path) -> Path:
    """Return the JSON sidecar path for the given model file."""
    return model_path.with_suffix(".json")


def _read_image_size(model_path: Path) -> int:
    """Return image_size from best_model.json sidecar, falling back to 96."""
    sidecar = _sidecar_path(model_path)
    if sidecar.exists():
        with sidecar.open() as fh:
            meta = json.load(fh)
        # sidecar may store it at top level or nested under a config key
        size = meta.get("image_size") or meta.get("config", {}).get("image_size")
        if size is not None:
            return int(size)
    return 96  # safe fallback — matches original PCam patch size


def _preprocess(img_path: Path, image_size: int) -> np.ndarray:
    """Load a PNG, resize to image_size×image_size, normalise to [0, 1]."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((image_size, image_size))
    return np.array(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_model():
    """Load the model once per test module and share across tests."""
    path = _model_path()
    if not path.exists():
        pytest.skip(f"Model not found at {path} — set BEST_MODEL_PATH to run inference tests.")
    return tf.keras.models.load_model(str(path))


@pytest.fixture(scope="module")
def image_size():
    """Return the image size expected by the deployed model."""
    return _read_image_size(_model_path())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelLoad:
    """Test model loading."""

    def test_load_model(self):
        """Load model from BEST_MODEL_PATH and verify it has a predict method."""
        path = _model_path()
        assert path.exists(), (
            f"Model file not found at {path}. "
            "Set BEST_MODEL_PATH or place model in artifacts/models/"
        )
        model = tf.keras.models.load_model(str(path))
        assert model is not None
        assert hasattr(model, "predict")

    def test_sidecar_exists(self):
        """Verify the best_model.json sidecar is present alongside the model."""
        path = _model_path()
        if not path.exists():
            pytest.skip("Model not found — skipping sidecar check.")
        sidecar = _sidecar_path(path)
        assert sidecar.exists(), (
            f"Sidecar {sidecar} is missing. "
            "Upload best_model.json together with best_model.keras."
        )

    def test_sidecar_contains_image_size(self):
        """Verify the sidecar reports a plausible image_size."""
        path = _model_path()
        sidecar = _sidecar_path(path)
        if not path.exists() or not sidecar.exists():
            pytest.skip("Model or sidecar not found.")
        size = _read_image_size(path)
        assert size in (32, 64, 96, 128, 160, 224, 256), (
            f"Unexpected image_size {size} in sidecar — check best_model.json."
        )


class TestModelInference:
    """Test model inference on example images."""

    def test_inference_on_example_image(self, loaded_model, image_size):
        """Resize example image to model's native size and verify a valid probability."""
        example_path = Path("backend/src/examples/cancer/cancer_01.png")
        assert example_path.exists(), f"Example image not found at {example_path}"

        img_array = _preprocess(example_path, image_size)
        img_batch = np.expand_dims(img_array, axis=0)

        output = loaded_model.predict(img_batch, verbose=0)
        score = float(output[0, 0])

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Probability should be in [0, 1], got {score}"

    def test_inference_batch(self, loaded_model, image_size):
        """Verify batch inference works on two images of the model's native size."""
        cancer_path = Path("backend/src/examples/cancer/cancer_01.png")
        normal_path = Path("backend/src/examples/no_cancer/no_cancer_01.png")

        assert cancer_path.exists(), f"Example image not found at {cancer_path}"
        assert normal_path.exists(), f"Example image not found at {normal_path}"

        img_batch = np.stack(
            [_preprocess(p, image_size) for p in (cancer_path, normal_path)],
            axis=0,
        )

        outputs = loaded_model.predict(img_batch, verbose=0)

        assert outputs.shape[0] == 2, f"Expected 2 outputs, got {outputs.shape[0]}"
        for i, score in enumerate(outputs[:, 0]):
            assert 0.0 <= float(score) <= 1.0, (
                f"Probability {i} should be in [0, 1], got {score}"
            )
