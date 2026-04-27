#!/usr/bin/env python3
"""Download best_model.keras from GCS during Docker build time.

Reads MODEL_BUCKET_NAME and BEST_MODEL_PATH from environment variables.
Downloads the model file from gs://{bucket}/best_model.keras to the specified local path.
"""

import os
import sys
from google.cloud import storage


def download_model():
    """Download model from GCS if bucket name is provided."""
    bucket_name = os.environ.get("MODEL_BUCKET_NAME")
    local_path = os.environ.get("BEST_MODEL_PATH", "/app/artifacts/models/best_model.keras")
    model_blob = "best_model.keras"

    if not bucket_name:
        print("❌ ERROR: MODEL_BUCKET_NAME environment variable is not set.")
        print("   Set it as a build arg: docker build --build-arg MODEL_BUCKET_NAME=your-bucket ...")
        sys.exit(1)

    print(f"📥 Downloading model from gs://{bucket_name}/{model_blob}...")

    # Ensure target directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_blob)
        blob.download_to_filename(local_path)

        # Verify download
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✅ Model downloaded successfully: {local_path} ({file_size_mb:.2f} MB)")

    except Exception as e:
        print(f"❌ ERROR: Failed to download model from GCS: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
