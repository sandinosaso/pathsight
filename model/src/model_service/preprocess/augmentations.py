"""Light augmentations for histopathology (avoid heavy color jitter)."""

from __future__ import annotations

import tensorflow as tf

def augment_train_image(image: tf.Tensor) -> tf.Tensor:
    """Apply light spatial and colour augmentations to a single image tensor.

    Augmentations are intentionally mild to preserve the haematoxylin-eosin
    stain signal that the classifier relies on:
      - Random horizontal and vertical flips: orientation is arbitrary in
        histopathology slides, so all four 90-degree reflections are valid.
      - Random brightness jitter (±8 %): accounts for slide-to-slide
        illumination variation without washing out stain contrast.
      - Random contrast jitter (±8 %): similarly captures scanner-to-scanner
        dynamic-range differences.
    A final clip to [0, 1] ensures pixel values stay in the expected range
    after the numeric perturbations.

    Parameters
    ----------
    image:
        Float32 tensor of shape (H, W, 3) with values in [0, 1].

    Returns
    -------
    tf.Tensor
        Augmented float32 tensor of the same shape, clipped to [0, 1].
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.92, upper=1.08)
    return tf.clip_by_value(image, 0.0, 1.0)


def augment_pair(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Augment an (image, label) pair as used inside a tf.data.Dataset.map() call.

    Applies :func:`augment_train_image` to the image and passes the label
    through unchanged.  This signature matches the two-argument form expected
    by ``dataset.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)``.

    Parameters
    ----------
    image:
        Float32 tensor of shape (H, W, 3) with values in [0, 1].
    label:
        Integer scalar tensor (0 = normal, 1 = metastatic).

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
        ``(augmented_image, label)`` with the label unmodified.
    """
    return augment_train_image(image), label
