"""Build batched `tf.data` pipelines for PCam."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_datasets as tfds

from model_service.config import ModelServiceConfig
from model_service.preprocess.tfds_pcam_loader import load_pcam_splits
from model_service.preprocess.augmentations import augment_pair
from model_service.preprocess.transforms import apply_resize_normalize, preprocess_for


def _preprocess_image(
    image: tf.Tensor,
    label: tf.Tensor,
    *,
    image_size: int,
    preprocess_mode: str,
    augment: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Resize, normalise to [0,1], optionally augment, then apply backbone preprocessing.

    Parameters
    ----------
    image_size:
        Target spatial dimension (square).
    preprocess_mode:
        One of ``"efficientnet"``, ``"mobilenetv3"``, ``"resnet"``,
        ``"convnext"``, or ``"none"``.  **Required** — no default, so a
        forgotten keyword argument surfaces as a TypeError rather than silently
        feeding wrong-range values to the model.
    augment:
        Whether to apply random augmentations (train only).
    """
    image, label = apply_resize_normalize(image, label, image_size=image_size)

    if augment:
        image, label = augment_pair(image, label)

    if preprocess_mode != "none":
        image = preprocess_for(preprocess_mode, image)

    return image, label


def _balanced_subset(ds: tf.data.Dataset, n_total: int, seed: int = 42) -> tf.data.Dataset:
    """Return a random balanced subset of *n_total* samples.

    Uses :func:`tf.data.Dataset.rejection_resample` (a built-in tf.data op) to
    enforce a 50/50 class split, then shuffles and limits to ``n_total``.
    PCam labels are integer scalars (0 = normal, 1 = metastatic).

    A large upstream shuffle ensures the resampler draws from the full
    distribution rather than a consecutive slice of the TFRecord shards.
    """
    # Buffer size: hold exactly n_total items so rejection_resample sees the
    # full subset distribution, but cap at 50 000 to avoid OOM on machines
    # with limited RAM (262 144 × 27 KB ≈ 7 GB — way too much for a shuffle).
    # For the small subsets used here (20k–50k) this gives a perfect shuffle
    # because the entire subset fits in the buffer.
    ds = ds.shuffle(buffer_size=min(n_total, 50_000), seed=seed)
    ds = ds.rejection_resample(
        class_func=lambda _x, y: tf.cast(y, tf.int32),
        target_dist=[0.5, 0.5],
        seed=seed,
    )
    # rejection_resample wraps each element as (class_id, original_element)
    ds = ds.map(lambda _class_id, xy: xy)
    return ds.take(n_total)


def build_pcam_datasets(
    *,
    backbone: str,
    image_size: int | None = None,
    data_dir: str | None = None,
    download: bool = True,
    max_train_samples: int | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Return (train, val, test, info) batched and prefetched datasets.

    Parameters
    ----------
    backbone:
        Backbone name (e.g. ``"convnexttiny"``, ``"efficientnetb0"``,
        ``"baseline"``).  Determines the preprocessing family applied to every
        image so training and inference always use the same transformation.
    image_size:
        Override ``config.data.image_size`` for a single run without touching
        env vars.  Useful when sweeping multiple sizes in one Python session.
    data_dir:
        Override the TFDS data directory.
    download:
        Whether to download the dataset if not present.
    max_train_samples:
        If set, limit training to this many samples using a balanced random
        subset (50/50 per class).  Val and test are scaled proportionally
        at 1/5 of ``max_train_samples`` each.  Pass ``None`` to use the full
        dataset (~262 k train / 32 k val / 32 k test).
    """
    from model_service.training.backbones import preprocess_mode as _mode_for

    dc = ModelServiceConfig().data
    autotune = tf.data.AUTOTUNE

    effective_size = image_size if image_size is not None else dc.image_size
    # Resolve mode once here — closures below capture the resolved string, not
    # the parameter, so there is no risk of the legacy-flag closure bug.
    mode = _mode_for(backbone)

    train_raw, val_raw, test_raw, info = load_pcam_splits(data_dir=data_dir, download=download)

    if max_train_samples is not None:
        n_eval = max(max_train_samples // 5, 128)
        train_raw = _balanced_subset(train_raw, max_train_samples, seed=dc.seed)
        val_raw = _balanced_subset(val_raw, n_eval, seed=dc.seed + 1)
        test_raw = _balanced_subset(test_raw, n_eval, seed=dc.seed + 2)

    def map_train(x, y):
        return _preprocess_image(
            x, y,
            image_size=effective_size,
            preprocess_mode=mode,
            augment=dc.augment_train,
        )

    def map_eval(x, y):
        return _preprocess_image(
            x, y,
            image_size=effective_size,
            preprocess_mode=mode,
            augment=False,
        )

    shuffle_buf = min(dc.shuffle_buffer, max_train_samples) if max_train_samples else dc.shuffle_buffer
    train_ds = train_raw.map(map_train, num_parallel_calls=autotune)
    if max_train_samples is not None:
        # Without repeat(), Keras exhausts the finite subset after epoch 1 and
        # stops training early.  Augmentation is applied fresh on each pass so
        # the model still sees different augmented views per epoch.
        train_ds = train_ds.repeat()
    train_ds = (
        train_ds
        .shuffle(shuffle_buf, seed=dc.seed)
        .batch(dc.batch_size)
        .prefetch(autotune)
    )

    val_ds = val_raw.map(map_eval, num_parallel_calls=autotune).batch(dc.batch_size)
    test_ds = test_raw.map(map_eval, num_parallel_calls=autotune).batch(dc.batch_size)

    if dc.cache:
        val_ds = val_ds.cache()
        test_ds = test_ds.cache()

    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, info
