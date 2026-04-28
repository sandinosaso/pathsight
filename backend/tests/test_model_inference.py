"""Smoke tests for model loading and inference."""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


class TestModelLoad:
    """Test model loading."""

    def test_load_model(self):
        """Load model from BEST_MODEL_PATH and verify it exists."""
        # Get model path from environment or use default
        model_path = os.getenv("BEST_MODEL_PATH", "artifacts/models/best_model.keras")

        # Check file exists
        assert Path(model_path).exists(), f"Model file not found at {model_path}"

        # Try to load the model
        model = tf.keras.models.load_model(model_path)
        assert model is not None, "Model should not be None after loading"
        assert hasattr(model, "predict"), "Model should have predict method"


class TestModelInference:
    """Test model inference on example images."""

    def test_inference_on_example_image(self):
        """Load example PNG, run inference, verify output is valid probability."""
        # Get model path from environment or use default
        model_path = os.getenv("BEST_MODEL_PATH", "artifacts/models/best_model.keras")

        # Skip test if model doesn't exist
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Set BEST_MODEL_PATH or place model in artifacts/models/")

        # Load the model
        model = tf.keras.models.load_model(model_path)

        # Load an example PNG image
        example_image_path = Path("backend/src/examples/cancer/cancer_01.png")
        assert example_image_path.exists(), f"Example image not found at {example_image_path}"

        # Read and preprocess the image
        img = Image.open(example_image_path).convert("RGB")
        img_resized = img.resize((96, 96))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0  # Normalize to [0, 1]

        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)

        # Run inference
        output = model.predict(img_batch, verbose=0)
        score = float(output[0, 0])

        # Verify output is a valid probability
        assert isinstance(score, float), f"Output should be float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Probability should be in [0.0, 1.0], got {score}"

    def test_inference_batch(self):
        """Verify batch inference works on multiple images."""
        # Get model path from environment or use default
        model_path = os.getenv("BEST_MODEL_PATH", "artifacts/models/best_model.keras")

        # Skip test if model doesn't exist
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Set BEST_MODEL_PATH or place model in artifacts/models/")

        # Load the model
        model = tf.keras.models.load_model(model_path)

        # Load 2 example images
        cancer_img_path = Path("backend/src/examples/cancer/cancer_01.png")
        normal_img_path = Path("backend/src/examples/no_cancer/no_cancer_01.png")

        assert cancer_img_path.exists(), f"Example image not found at {cancer_img_path}"
        assert normal_img_path.exists(), f"Example image not found at {normal_img_path}"

        # Read and preprocess both images
        images = []
        for img_path in [cancer_img_path, normal_img_path]:
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((96, 96))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            images.append(img_array)

        # Stack into batch
        img_batch = np.stack(images, axis=0)

        # Run inference on batch
        outputs = model.predict(img_batch, verbose=0)

        # Verify we get 2 outputs, each a valid probability
        assert outputs.shape[0] == 2, f"Should have 2 outputs, got {outputs.shape[0]}"
        for i, score in enumerate(outputs[:, 0]):
            score_val = float(score)
            assert 0.0 <= score_val <= 1.0, f"Probability {i} should be in [0.0, 1.0], got {score_val}"
