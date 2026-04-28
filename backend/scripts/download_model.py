#!/usr/bin/env python3
"""Download best_model.keras and its metadata sidecar from GCS.

Reads MODEL_BUCKET_NAME and BEST_MODEL_PATH from environment variables.
Downloads:
  gs://{bucket}/best_model.keras → local BEST_MODEL_PATH
  gs://{bucket}/best_model.json  → same directory, .json extension
"""

import os
import sys
from pathlib import Path

from google.cloud import storage


def download_model():
    """Download model and sidecar from GCS."""
    bucket_name = os.environ.get("MODEL_BUCKET_NAME")
    local_model_path = os.environ.get("BEST_MODEL_PATH", "/app/artifacts/models/best_model.keras")

    if not bucket_name:
        print("ERROR: MODEL_BUCKET_NAME environment variable is not set.")
        sys.exit(1)

    local_sidecar_path = str(Path(local_model_path).with_suffix(".json"))

    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        print(f"Downloading model from gs://{bucket_name}/best_model.keras ...")
        bucket.blob("best_model.keras").download_to_filename(local_model_path)
        size_mb = os.path.getsize(local_model_path) / (1024 * 1024)
        print(f"  Model saved to {local_model_path} ({size_mb:.2f} MB)")

        print(f"Downloading sidecar from gs://{bucket_name}/best_model.json ...")
        bucket.blob("best_model.json").download_to_filename(local_sidecar_path)
        print(f"  Sidecar saved to {local_sidecar_path}")

    except Exception as e:
        print(f"ERROR: Failed to download from GCS: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
