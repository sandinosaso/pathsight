import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from model.src.model_service.config import ModelServiceConfig


@dataclass
class LoadedModel:
    """Bundle returned by load_model_trained().

    Carries the Keras model together with the metadata that drives
    preprocessing at inference time, so callers never have to re-derive
    backbone or image_size from the filename or environment.
    """
    model: tf.keras.Model
    backbone: str
    image_size: int
    preprocess_mode: str


def load_model_trained() -> LoadedModel:
    config = ModelServiceConfig()
    model_path = os.path.normpath(config.data.best_model_path)
    sidecar_path = os.path.splitext(model_path)[0] + ".json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(
            f"Model metadata sidecar not found at: {sidecar_path}. "
            "Each .keras file must be accompanied by a JSON sidecar describing "
            "its backbone and image_size. "
            "Upload one with: "
            'echo \'{"backbone": "convnexttiny", "image_size": 96}\' | '
            "gcloud storage cp - gs://<bucket>/best_model.json"
        )

    meta = json.loads(Path(sidecar_path).read_text())
    backbone = meta["backbone"]
    image_size = int(meta["image_size"])

    from model.src.model_service.training.backbones import preprocess_mode as _mode_for
    mode = _mode_for(backbone)

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    file_modified_date = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Loading model from: {model_path}")
    print(f"  backbone={backbone}  image_size={image_size}  preprocess_mode={mode}")
    print(f"  size={file_size_mb:.2f} MB  modified={file_modified_date}")

    return LoadedModel(
        model=tf.keras.models.load_model(model_path),
        backbone=backbone,
        image_size=image_size,
        preprocess_mode=mode,
    )


def predict_logic(model: tf.keras.Model, img_data: tf.Tensor) -> float:
    img_data = tf.expand_dims(img_data, axis=0)
    prediction = model.predict(img_data, verbose=0)
    result = float(prediction[0][0])
    print(f"Result predicted: {result:.4f}")
    return result
