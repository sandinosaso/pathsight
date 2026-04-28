"""H&E-tuned augmentations for histopathology patches.

Recipe based on Tellez et al. 2019 ("Quantifying the effects of data
augmentation and stain color normalization in convolutional neural networks
for computational pathology").  The defaults below match the values that
study found optimal for PCam-class problems; all knobs are exposed as
keyword arguments and wired through ``ModelServiceConfig.data`` so
experiments can iterate via env vars without code changes.
"""

from __future__ import annotations

import tensorflow as tf


def augment_train_image(
    image: tf.Tensor,
    *,
    brightness_delta: float = 0.15,
    contrast_delta: float = 0.15,
    saturation_delta: float = 0.15,
    hue_delta: float = 0.04,
    zoom_min_area: float = 0.9,
    use_rot90: bool = True,
) -> tf.Tensor:
    """Apply the H&E augmentation recipe to a single image.

    Augmentations applied (in order):

    1. **Random 90-degree rotation** (if ``use_rot90``) — slides have no
       canonical orientation, so all four rotations are valid samples.
    2. **Random horizontal + vertical flips** — combined with rot90 this
       gives the full D4 dihedral group (8 distinct orientations).
    3. **Brightness jitter** ``[-brightness_delta, +brightness_delta]`` —
       captures slide-to-slide illumination variation.
    4. **Contrast jitter** in ``[1-contrast_delta, 1+contrast_delta]`` —
       captures scanner dynamic-range differences.
    5. **Saturation jitter** in ``[1-saturation_delta, 1+saturation_delta]``
       — captures stain-intensity variation across labs.
    6. **Hue jitter** ``[-hue_delta, +hue_delta]`` — captures scanner /
       staining-protocol colour shifts.
    7. **Random zoom-in** (if ``zoom_min_area < 1.0``) — random crop
       retaining at least ``zoom_min_area`` of the original area, then
       resized back to the input shape.  Keeps the central diagnostic
       region intact while adding scale invariance.
    8. **Clip to [0, 1]** to undo any numeric overshoot from the jitters.

    Parameters
    ----------
    image:
        Float32 tensor, shape ``(H, W, 3)``, values in ``[0, 1]``.
    brightness_delta, contrast_delta, saturation_delta, hue_delta:
        Strengths of the four colour-space jitters.  Defaults match
        Tellez 2019.  Set 0.0 to disable a specific jitter.
    zoom_min_area:
        Minimum fraction of the original area kept by the random crop
        (1.0 disables zoom).  0.9 corresponds to up to ~5% linear
        zoom-in, which preserves the central 32x32 diagnostic region.
    use_rot90:
        Whether to apply random 90-degree rotations.

    Returns
    -------
    tf.Tensor
        Augmented float32 image with the same shape as the input,
        clipped to ``[0, 1]``.
    """
    if use_rot90:
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if brightness_delta > 0.0:
        image = tf.image.random_brightness(image, max_delta=brightness_delta)
    if contrast_delta > 0.0:
        image = tf.image.random_contrast(
            image, lower=1.0 - contrast_delta, upper=1.0 + contrast_delta
        )
    if saturation_delta > 0.0:
        image = tf.image.random_saturation(
            image, lower=1.0 - saturation_delta, upper=1.0 + saturation_delta
        )
    if hue_delta > 0.0:
        image = tf.image.random_hue(image, max_delta=hue_delta)

    if zoom_min_area < 1.0:
        # Random crop with min-area constraint, then resize back.
        # Linear scale = sqrt(area_fraction) so the area stays >= zoom_min_area.
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        scale = tf.random.uniform([], minval=tf.sqrt(tf.constant(zoom_min_area, tf.float32)), maxval=1.0)
        new_h = tf.maximum(tf.cast(tf.cast(h, tf.float32) * scale, tf.int32), 1)
        new_w = tf.maximum(tf.cast(tf.cast(w, tf.float32) * scale, tf.int32), 1)
        image = tf.image.random_crop(image, size=[new_h, new_w, 3])
        image = tf.image.resize(image, (h, w), method="bicubic")

    return tf.clip_by_value(image, 0.0, 1.0)


def augment_pair(
    image: tf.Tensor,
    label: tf.Tensor,
    **aug_kwargs,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Augment an ``(image, label)`` pair as used inside a ``tf.data.Dataset.map()`` call.

    Forwards ``aug_kwargs`` to :func:`augment_train_image`.  The label is
    passed through unchanged.  The two-argument signature (modulo
    ``**aug_kwargs``) matches what ``dataset.map(augment_pair, ...)``
    expects.
    """
    return augment_train_image(image, **aug_kwargs), label
