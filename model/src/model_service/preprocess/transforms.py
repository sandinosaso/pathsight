"""Resize and normalization."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

def resize_image(image: tf.Tensor, size: int) -> tf.Tensor:
    return tf.image.resize(image, (size, size), method="bilinear")


def to_float01(image: tf.Tensor) -> tf.Tensor:
    return tf.cast(image, tf.float32) / 255.0


def apply_resize_normalize(
    image: tf.Tensor,
    label: tf.Tensor,
    *,
    image_size: int
) -> tuple[tf.Tensor, tf.Tensor]:
    image = resize_image(image, image_size)
    image = to_float01(image)

    label_f = tf.cast(label, tf.float32)
    return image, label_f
