"""Unified backbone factory for PCam transfer-learning experiments.

Supported backbones
-------------------
efficientnetb0   — EfficientNetB0  (~4 M params)
mobilenetv3small — MobileNetV3Small (~2.5 M params)
mobilenetv3large — MobileNetV3Large (~5.4 M params)
resnet50         — ResNet50         (~25 M params)
convnexttiny     — ConvNeXtTiny     (~28 M params)
baseline         — from-scratch baseline CNN (no ImageNet backbone)

Each transfer-learning backbone wraps a frozen ImageNet backbone with the
same small classification head so results are directly comparable across runs.

Preprocessing notes
-------------------
Each backbone family expects a different pixel-value range:
  efficientnet  — [0, 255]  (handled by EfficientNet's internal Rescaling)
  mobilenetv3   — [-1, 1]   (handled by MobileNetV3's internal Rescaling)
  resnet        — ImageNet channel mean subtracted, [0, 255] input
  convnext      — ImageNet mean/std normalised, [0, 1] input
  baseline      — [0, 1] float32 (no extra normalisation needed)

Use `preprocess_mode(backbone)` to get the string key understood by
`transforms.preprocess_for()` so the data pipeline applies the right
preprocessing before feeding images to the model.
"""

from __future__ import annotations

from typing import Literal

from tensorflow import keras
from tensorflow.keras import layers

from model_service.config import ModelServiceConfig

BackboneName = Literal[
    "efficientnetb0",
    "mobilenetv3small",
    "mobilenetv3large",
    "resnet50",
    "convnexttiny",
    "baseline",
    "baseline_cnn",
]

PreprocessMode = Literal["efficientnet", "mobilenetv3", "resnet", "convnext", "none"]


def preprocess_mode(backbone: str) -> PreprocessMode:
    """Return the preprocessing key for the given backbone family.

    Every backbone name must resolve to one of the PreprocessMode literals.
    Raises ValueError for unknown names so misconfiguration surfaces at
    import/startup time rather than silently at inference.
    """
    if backbone.startswith("efficientnet"):
        return "efficientnet"
    if backbone.startswith("mobilenetv3"):
        return "mobilenetv3"
    if backbone.startswith("resnet"):
        return "resnet"
    if backbone.startswith("convnext"):
        return "convnext"
    if backbone in ("baseline", "baseline_cnn"):
        return "none"
    raise ValueError(f"Unknown backbone: {backbone!r}")


HeadStyle = Literal["default", "minimal"]
"""Head architecture variants.

``"default"``
    GAP → Dense(head_units, relu) → Dropout → Dense(1, sigmoid).
    More capacity; good with augmentation and large datasets.

``"minimal"``
    GAP → BatchNormalization → Dropout → Dense(1, sigmoid).
    No hidden Dense layer.  Matches the Shayan-notebook recipe
    (``base_model → GAP → BN → Dropout(0.3) → Dense(1, sigmoid)``).
    Fewer parameters; tends to generalise better on small datasets
    and when no augmentation is applied.
"""


def _build_head(
    x: keras.KerasTensor,
    head_units: int,
    head_dropout: float,
    head_style: HeadStyle = "default",
) -> keras.KerasTensor:
    """Build the classification head on top of the backbone's feature map.

    Parameters
    ----------
    head_style:
        ``"default"`` — GAP → Dense(head_units, relu) → Dropout → Dense(1, sigmoid).
        ``"minimal"`` — GAP → BatchNormalization → Dropout → Dense(1, sigmoid).
    """
    x = layers.GlobalAveragePooling2D()(x)
    if head_style == "minimal":
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(head_dropout)(x)
    else:
        x = layers.Dense(head_units, activation="relu")(x)
        x = layers.Dropout(head_dropout)(x)
    return layers.Dense(1, activation="sigmoid")(x)


# String -> Keras metric factory.  Single source of truth for what each
# metric name in ``ModelServiceConfig.train.metrics`` (or the MODEL_METRICS
# env var) maps to.  Add a new entry here to expose a new metric to runs
# without changing any other code.
_METRIC_FACTORY = {
    "accuracy":  lambda: keras.metrics.BinaryAccuracy(name="accuracy"),
    "precision": lambda: keras.metrics.Precision(name="precision"),
    "recall":    lambda: keras.metrics.Recall(name="recall"),
    "auc":       lambda: keras.metrics.AUC(name="auc", curve="ROC"),
    "pr_auc":    lambda: keras.metrics.AUC(name="pr_auc", curve="PR"),
}


def _build_metrics() -> list[keras.metrics.Metric]:
    """Build the metric list from ``ModelServiceConfig().train.metrics``.

    Unknown keys are silently skipped — the env var stays the single source
    of truth and a typo there only loses that metric, never breaks training.
    """
    cfg = ModelServiceConfig().train
    metrics: list[keras.metrics.Metric] = []
    for name in cfg.metrics:
        factory = _METRIC_FACTORY.get(name)
        if factory is not None:
            metrics.append(factory())
    return metrics


def _compile(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=_build_metrics(),
    )
    return model


def build_transfer_model(
    backbone: BackboneName,
    input_shape: tuple[int, int, int],
    *,
    learning_rate: float = 1e-4,
    weights: str = "imagenet",
    head_dropout: float = 0.3,
    head_units: int = 128,
    head_style: HeadStyle = "default",
    freeze_backbone: bool = True,
) -> keras.Model:
    """Build a transfer-learning model for binary PCam classification.

    Parameters
    ----------
    backbone:
        One of the BackboneName literals.
    input_shape:
        HxWxC tuple, e.g. ``(224, 224, 3)``.
    learning_rate:
        Adam LR for stage-1 (head-only) training.
    weights:
        ``"imagenet"`` or a path to pre-trained weights.
    head_dropout:
        Dropout rate applied before the final sigmoid unit.
    head_units:
        Width of the hidden Dense layer (``head_style="default"`` only).
    head_style:
        ``"default"`` — GAP → Dense(head_units) → Dropout → sigmoid.
        ``"minimal"`` — GAP → BN → Dropout → sigmoid (no hidden Dense).
    freeze_backbone:
        When ``True`` (default) the backbone is frozen at construction time
        so only the head trains in stage 1.  Set to ``False`` to train the
        full network end-to-end from epoch 1 (matches the Shayan-notebook
        recipe where ``base_model.trainable = True`` from the start).
    """
    inputs = keras.Input(shape=input_shape)

    if backbone == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetV2B0
        base = EfficientNetV2B0(include_top=False, weights=weights)
        base.trainable = not freeze_backbone
        # training=True: use batch statistics so BN calibrates to PCam colours
        x = base(inputs, training=True)

    elif backbone == "mobilenetv3small":
        from tensorflow.keras.applications import MobileNetV3Small
        base = MobileNetV3Small(include_top=False, weights=weights, minimalistic=False)
        base.trainable = not freeze_backbone
        x = base(inputs, training=True)

    elif backbone == "mobilenetv3large":
        from tensorflow.keras.applications import MobileNetV3Large
        base = MobileNetV3Large(include_top=False, weights=weights, minimalistic=False)
        base.trainable = not freeze_backbone
        x = base(inputs, training=True)

    elif backbone == "resnet50":
        from tensorflow.keras.applications import ResNet50
        base = ResNet50(include_top=False, weights=weights)
        base.trainable = not freeze_backbone
        x = base(inputs, training=True)

    elif backbone == "convnexttiny":
        from tensorflow.keras.applications import ConvNeXtTiny
        base = ConvNeXtTiny(include_top=False, weights=weights)
        base.trainable = not freeze_backbone
        x = base(inputs, training=True)

    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")

    outputs = _build_head(x, head_units=head_units, head_dropout=head_dropout, head_style=head_style)
    model = keras.Model(inputs, outputs, name=f"{backbone}_pcam")
    return _compile(model, learning_rate)


def unfreeze_top(
    model: keras.Model,
    backbone: str,
    *,
    num_layers: int = 20,
    learning_rate: float = 1e-5,
) -> keras.Model:
    """Unfreeze the top ``num_layers`` of the backbone and recompile with low LR.

    Works for all backbone families by finding the first nested
    ``keras.Model`` layer (the backbone submodel).
    """
    base: keras.Model | None = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base = layer
            break
    if base is None:
        raise RuntimeError("Could not find a nested backbone model inside the outer model.")

    base.trainable = True
    n = len(base.layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= max(0, n - num_layers)

    return _compile(model, learning_rate)


def find_gradcam_layer(model: keras.Model) -> str:
    """Return the name of the last spatial conv layer suitable for Grad-CAM."""
    # Look inside the nested backbone submodel
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            backbone_layers = layer.layers
            # Walk backwards; first Conv2D found is the last spatial feature map
            for bl in reversed(backbone_layers):
                if isinstance(bl, layers.Conv2D):
                    return bl.name
            # Some backbones (e.g. ConvNeXt) use DepthwiseConv2D
            for bl in reversed(backbone_layers):
                if isinstance(bl, layers.DepthwiseConv2D):
                    return bl.name
    raise RuntimeError("No suitable Grad-CAM layer found in model.")
