"""Build batched `tf.data` pipelines for PCam."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_datasets as tfds

from model_service.config import ModelServiceConfig
# from model_service.preprocess.augmentations import augment_pair
from model_service.preprocess.tfds_pcam_loader import load_pcam_splits
# from model_service.preprocess.transforms import apply_resize_normalize, preprocess_for


def _preprocess_image(
    image: tf.Tensor,
    label: tf.Tensor,
    *,
    image_size: int,
    augment: bool,
    preprocess_mode: str = "none",
    stain_normalise: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    # # Step 0 (optional): Macenko stain normalisation runs on the raw uint8
    # # tensor *before* resize/float-cast so the stain decomposition operates
    # # on the original pixel values.
    # # Step 1: resize and cast to float [0, 1]. Backbone preprocessing is
    # # intentionally deferred so augmentation always operates on the [0, 1]
    # # range that augment_pair expects (it clips to [0, 1] at the end).
    # # Calling preprocess_for *before* augmentation would push pixel values
    # # to [0, 255] (or [-1, 1]), causing the final clip to destroy the signal.
    # image, label = apply_resize_normalize(
    #     image,
    #     label,
    #     image_size=image_size,
    #     use_efficientnet_preprocess=False,  # always False here; applied below
    #     stain_normalise=stain_normalise,
    # )

    # # Step 2: augment while the image is still in [0, 1].
    # if augment:
    #     image, label = augment_pair(image, label)

    # # Step 3: apply backbone-specific preprocessing (scales out of [0,1]).
    # if preprocess_mode and preprocess_mode != "none":
    #     image = preprocess_for(preprocess_mode, image)

    return image, label


def _balanced_subset(ds: tf.data.Dataset, n_total: int, seed: int = 42) -> tf.data.Dataset:
    """Return a random balanced subset of *n_total* samples.

    Uses :func:`tf.data.Dataset.rejection_resample` (a built-in tf.data op) to
    enforce a 50/50 class split, then shuffles and limits to ``n_total``.
    PCam labels are integer scalars (0 = normal, 1 = metastatic).

    A large upstream shuffle ensures the resampler draws from the full
    distribution rather than a consecutive slice of the TFRecord shards.
    """
    ds = ds.shuffle(buffer_size=min(n_total * 8, 262_144), seed=seed)
    ds = ds.rejection_resample(
        class_func=lambda _x, y: tf.cast(y, tf.int32),
        target_dist=[0.5, 0.5],
        seed=seed,
    )
    # rejection_resample wraps each element as (class_id, original_element)
    ds = ds.map(lambda _class_id, xy: xy)
    return ds.take(n_total)


def build_pcam_datasets(
    config: ModelServiceConfig | None = None,
    *,
    data_dir: str | None = None,
    download: bool = True,
    # Legacy flag kept for backward compatibility with existing training scripts.
    use_efficientnet_preprocess: bool = False,
    # Per-run overrides (used by the benchmark framework).
    image_size: int | None = None,
    preprocess_mode: str | None = None,
    max_train_samples: int | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Return (train, val, test, info) batched and prefetched datasets.

    Parameters
    ----------
    config:
        ``ModelServiceConfig`` instance. Defaults to a fresh one from env vars.
    data_dir:
        Override the TFDS data directory.
    download:
        Whether to download the dataset if not present.
    use_efficientnet_preprocess:
        Legacy kwarg.  Prefer ``preprocess_mode="efficientnet"`` instead.
    image_size:
        Override ``config.data.image_size`` for a single run without touching
        env vars.  Useful when sweeping multiple sizes in one Python session.
    preprocess_mode:
        Override the preprocessing family.  One of ``"efficientnet"``,
        ``"mobilenetv3"``, ``"resnet"``, ``"convnext"``, ``"none"``.
        Defaults to ``"efficientnet"`` when ``use_efficientnet_preprocess``
        is ``True``, otherwise ``"none"``.
    max_train_samples:
        If set, limit training to this many samples using a balanced random
        subset (50/50 per class).  Val and test are scaled proportionally
        at 1/5 of ``max_train_samples`` each.  Pass ``None`` to use the full
        dataset (~262 k train / 32 k val / 32 k test).
    """
    if config is None:
        config = ModelServiceConfig()
    dc = config.data
    autotune = tf.data.AUTOTUNE

    # Resolve effective image size and preprocess mode
    effective_size = image_size if image_size is not None else dc.image_size

    if preprocess_mode is not None:
        effective_mode = preprocess_mode
    elif use_efficientnet_preprocess:
        effective_mode = "efficientnet"
    else:
        effective_mode = "none"

    train_raw, val_raw, test_raw, info = load_pcam_splits(data_dir=data_dir, download=download)

    if max_train_samples is not None:
        n_eval = max(max_train_samples // 5, 128)
        train_raw = _balanced_subset(train_raw, max_train_samples, seed=dc.seed)
        val_raw = _balanced_subset(val_raw, n_eval, seed=dc.seed + 1)
        test_raw = _balanced_subset(test_raw, n_eval, seed=dc.seed + 2)

    def map_train(x, y):
        return _preprocess_image(
            x,
            y,
            image_size=effective_size,
            augment=dc.augment_train,
            preprocess_mode=effective_mode,
            stain_normalise=dc.stain_normalise,
        )

    def map_eval(x, y):
        return _preprocess_image(
            x,
            y,
            image_size=effective_size,
            augment=False,
            preprocess_mode=effective_mode,
            stain_normalise=dc.stain_normalise,
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

    # cache → prefetch: read data once, then serve from memory fast
    val_ds_with_preprocess = val_raw.map(map_eval, num_parallel_calls=autotune).batch(dc.batch_size)
    test_ds_with_preprocess = test_raw.map(map_eval, num_parallel_calls=autotune).batch(dc.batch_size)

    if dc.cache:
        val_ds_with_preprocess = val_ds_with_preprocess.cache()
        test_ds_with_preprocess = test_ds_with_preprocess.cache()

    val_ds_with_preprocess = val_ds_with_preprocess.prefetch(autotune)
    test_ds_with_preprocess = test_ds_with_preprocess.prefetch(autotune)

    return train_ds, val_ds_with_preprocess, test_ds_with_preprocess, info
