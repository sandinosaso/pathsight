"""Resize and normalization."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

def resize_image(image: tf.Tensor, size: int) -> tf.Tensor:
    return tf.image.resize(image, (size, size), method="bilinear")


def to_float01(image: tf.Tensor) -> tf.Tensor:
    return tf.cast(image, tf.float32) / 255.0


def efficientnet_preprocess(image: tf.Tensor) -> tf.Tensor:
    """ImageNet preprocessing for EfficientNet (expects float32 [0,1] input)."""
    from tensorflow.keras.applications.efficientnet import preprocess_input

    return preprocess_input(image * 255.0)


def preprocess_for(mode: str, image: tf.Tensor) -> tf.Tensor:
    """Apply backbone-family preprocessing to a float32 [0,1] image tensor.

    Parameters
    ----------
    mode:
        One of ``"efficientnet"``, ``"mobilenetv3"``, ``"resnet"``,
        ``"convnext"``, or ``"none"`` (returns image unchanged).
    image:
        Float32 tensor in [0, 1] range, shape (..., H, W, 3).
    """
    if mode == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input(image * 255.0)

    if mode == "mobilenetv3":
        # MobileNetV3 expects [0,255] input; its internal Rescaling layer
        # converts to [-1, 1] so we just undo the [0,1] normalisation.
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        return preprocess_input(image * 255.0)

    if mode == "resnet":
        # ResNet50 expects [0,255] with ImageNet channel-mean subtraction.
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(image * 255.0)

    if mode == "convnext":
        # ConvNeXtTiny expects [0,255] with ImageNet mean/std normalisation.
        from tensorflow.keras.applications.convnext import preprocess_input
        return preprocess_input(image * 255.0)

    if mode == "none":
        return image

    raise ValueError(f"Unknown preprocess mode: {mode!r}")


def _stain_normalise_tensor(image_tensor: tf.Tensor) -> tf.Tensor:
    """Macenko normalisation wrapped for use inside ``tf.py_function``.

    Accepts and returns uint8 tensors so we can call this before the
    float cast and resize steps.
    """
    from PIL import Image as _PILImage

    from model_service.preprocess.stain_normalisation import normalise_pil

    arr = image_tensor.numpy().astype(np.uint8)
    result = normalise_pil(_PILImage.fromarray(arr))
    return np.asarray(result, dtype=np.uint8)


def apply_resize_normalize(
    image: tf.Tensor,
    label: tf.Tensor,
    *,
    image_size: int,
    stain_normalise: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    if stain_normalise:
        image = tf.py_function(
            func=_stain_normalise_tensor,
            inp=[image],
            Tout=tf.uint8,
        )
        image.set_shape([None, None, 3])
    image = resize_image(image, image_size)
    image = to_float01(image)

    label_f = tf.cast(label, tf.float32)
    return image, label_f
