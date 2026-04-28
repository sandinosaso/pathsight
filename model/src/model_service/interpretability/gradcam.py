"""Grad-CAM heatmap generation for binary-classifier models."""

from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model_service.training.backbones import find_gradcam_layer

logger = logging.getLogger(__name__)


def _find_layer_recursive(model: keras.Model, name: str) -> keras.layers.Layer | None:
    """Search ``model`` (and any nested ``keras.Model`` submodels) for a layer
    whose ``name`` matches."""
    for layer in model.layers:
        if layer.name == name:
            return layer
        if isinstance(layer, keras.Model):
            hit = _find_layer_recursive(layer, name)
            if hit is not None:
                return hit
    return None


def _find_backbone(model: keras.Model) -> keras.Model | None:
    """Return the first nested ``keras.Model`` sub-layer of ``model``,
    which is the backbone in all our transfer-learning architectures."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    return None


def _build_head_model(model: keras.Model, backbone: keras.Model) -> keras.Model:
    """Return a Keras model that maps backbone output → final prediction.

    Built by wiring the outer model's head layers onto a fresh ``keras.Input``
    that matches the backbone's output shape.  This avoids any cross-graph
    tensor references between the backbone's internal graph and the outer
    model's graph.
    """
    backbone_out_shape = tuple(backbone.output.shape[1:])
    head_input = keras.Input(shape=backbone_out_shape)
    x = head_input
    # Collect head layers: every layer in the outer model that comes after
    # the backbone in topological order (skip InputLayer and the backbone itself).
    backbone_idx = next(i for i, l in enumerate(model.layers) if l is backbone)
    for layer in model.layers[backbone_idx + 1 :]:
        x = layer(x)
    return keras.Model(head_input, x, name="gradcam_head")


def compute_gradcam(
    model: keras.Model,
    image_batch: tf.Tensor,
    target_layer_name: str,
) -> np.ndarray:
    """Return a 2-D float32 Grad-CAM heatmap (H', W') for a single-image batch.

    Strategy
    --------
    For transfer-learning models (backbone nested inside outer model),
    building ``keras.Model(outer_input, [nested_layer.output, outer_output])``
    fails with a "graph disconnected" error because the nested layer's output
    tensor lives in a different graph namespace from the outer model.

    Instead we:

    1. Call ``backbone(image_batch)`` inside the tape to obtain the spatial
       feature map ``conv_out``.
    2. ``tape.watch(conv_out)`` — the tape now records every op that *uses*
       ``conv_out`` from this point forward.
    3. Call ``head_model(conv_out)`` so predictions flow *through* the watched
       tensor; the tape can therefore differentiate the loss back to it.
    4. ``tape.gradient(loss, conv_out)`` yields the feature-map gradients.
    5. Apply the standard Grad-CAM formula.
    """
    image_batch = tf.cast(image_batch, tf.float32)

    backbone = _find_backbone(model)

    if backbone is None:
        # Flat model — target layer is directly in the outer model graph.
        target_layer = _find_layer_recursive(model, target_layer_name)
        if target_layer is None:
            raise ValueError(f"Layer {target_layer_name!r} not found in model")
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(image_batch, training=False)
            tape.watch(conv_out)
            loss = preds[:, 0]
        grads = tape.gradient(loss, conv_out)
    else:
        # Transfer-learning model: run backbone → watch its output → run head.
        head_model = _build_head_model(model, backbone)

        with tf.GradientTape() as tape:
            # backbone output = spatial feature map, e.g. (1, 3, 3, 1280)
            conv_out = backbone(image_batch, training=False)
            # Tape records ops that USE conv_out from here onward.
            tape.watch(conv_out)
            # Predictions flow THROUGH conv_out → gradients are well-defined.
            preds = head_model(conv_out, training=False)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_out)

    if grads is None:
        raise ValueError(
            f"GradientTape returned None for layer {target_layer_name!r}. "
            "Check that the layer is on the differentiable path to the model output."
        )

    # Standard Grad-CAM formula (batch-aware)
    pooled = tf.reduce_mean(grads, axis=[1, 2])                            # (B, C)
    heatmap = tf.reduce_sum(
        conv_out * pooled[:, tf.newaxis, tf.newaxis, :], axis=-1
    )                                                                       # (B, H', W')
    heatmap = tf.nn.relu(heatmap)
    return heatmap[0].numpy().astype(np.float32)                            # (H', W')


def _safe_find_layer(model: keras.Model) -> str | None:
    """Return the best Grad-CAM layer name, or None on any failure."""
    try:
        return find_gradcam_layer(model)
    except Exception:  # noqa: BLE001
        return None


def _zero_heatmap(image_batch: tf.Tensor) -> np.ndarray:
    h, w = int(image_batch.shape[1]), int(image_batch.shape[2])
    return np.zeros((h, w), dtype=np.float32)


def try_generate_gradcam(
    model: keras.Model,
    image_batch: tf.Tensor,
    *,
    target_layer_name: str | None = None,
) -> tuple[np.ndarray, str | None]:
    """Best-effort Grad-CAM.

    Returns ``(heatmap, layer_used)``.  On any failure returns a flat zero
    heatmap and ``layer_used=None`` so the HTTP endpoint is never broken.
    """
    layer = target_layer_name or _safe_find_layer(model)
    if layer is None:
        logger.warning("No suitable Grad-CAM layer found; returning zero heatmap.")
        return _zero_heatmap(image_batch), None
    try:
        hm = compute_gradcam(model, image_batch, layer)
        logger.debug("Grad-CAM succeeded for layer %r, heatmap shape %s", layer, hm.shape)
        return hm, layer
    except Exception as exc:  # noqa: BLE001
        logger.warning("Grad-CAM failed for layer %r: %s", layer, exc)
        return _zero_heatmap(image_batch), None
