"""Load PatchCamelyon via TensorFlow Datasets."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_datasets as tfds

from model_service.constants import PCAM_DATASET_NAME

def load_pcam_splits(
    *,
    data_dir: str | None = None,
    download: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Return (train, validation, test) as supervised datasets and dataset info."""
    builder = tfds.builder(PCAM_DATASET_NAME, data_dir=data_dir)
    if download:
        builder.download_and_prepare()

    train_ds = builder.as_dataset(split="train", as_supervised=True, shuffle_files=True)
    val_ds = builder.as_dataset(split="validation", as_supervised=True, shuffle_files=False)
    test_ds = builder.as_dataset(split="test", as_supervised=True, shuffle_files=False)
    return train_ds, val_ds, test_ds, builder.info
