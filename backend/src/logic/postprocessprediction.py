"""Format model outputs for API consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from model_service.constants import LABEL_ALIASES, LABEL_NAMES
from model_service.interpretability.gradcam import try_generate_gradcam
from model_service.interpretability.overlays import (
    array_to_png_base64,
    blend_overlay,
    bytes_to_rgb_u8,
    heatmap_to_rgb_u8,
)


@dataclass
class FormattedPrediction:
    predicted_label_internal: str
    predicted_label_display: str
    confidence: float
    probabilities_internal: dict[str, float]
    probabilities_display: dict[str, float]


def format_binary_prediction(
    prob_positive: float,
    *,
    threshold: float = 0.5,
) -> FormattedPrediction:
    """Format a raw sigmoid score into labels, confidence, and probabilities.

    ``threshold`` controls the positive/negative decision boundary.  Pass the
    model's ``test_threshold`` from its summary JSON so the classification
    matches the operating point used during evaluation (typically optimised for
    F1 on the test set, not the naive 0.5).
    """
    p1 = float(np.clip(prob_positive, 1e-7, 1 - 1e-7))
    p0 = 1.0 - p1
    if p1 >= threshold:
        internal = LABEL_NAMES[1]
        conf = p1
    else:
        internal = LABEL_NAMES[0]
        conf = p0
    probs_i = {LABEL_NAMES[0]: p0, LABEL_NAMES[1]: p1}
    probs_d = {
        LABEL_ALIASES[LABEL_NAMES[0]]: p0,
        LABEL_ALIASES[LABEL_NAMES[1]]: p1,
    }
    display = LABEL_ALIASES[internal]
    return FormattedPrediction(
        predicted_label_internal=internal,
        predicted_label_display=display,
        confidence=conf,
        probabilities_internal=probs_i,
        probabilities_display=probs_d,
    )


def build_prediction_response(
    score: float,
    raw_bytes: bytes,
    batch: tf.Tensor,
    loaded_model: "LoadedModel",  # type: ignore[name-defined]  # avoids circular import
    model_path_name: str,
) -> "PredictionResponse":  # type: ignore[name-defined]
    """Assemble the full API response from a raw inference score.

    Responsibilities:
    - Format label, confidence, and probability dict.
    - Decode the original upload to RGB for display.
    - Generate the Grad-CAM heatmap and colour-blended overlay (best-effort).
    - Encode all three images as PNG base64 strings.
    - Return a populated PredictionResponse ready to serialise.
    """
    from backend.src.logic.predict import LoadedModel  # local import avoids circular dep
    from backend.src.schemas import PredictionMeta, PredictionResponse

    # Use the model's own test_threshold (derived from PR-curve optimisation during
    # benchmarking) so the decision boundary matches what was evaluated.
    # Falls back to 0.5 if the summary is missing the field (e.g. older models).
    threshold = float(loaded_model.summary.get("test_threshold", 0.5))
    fmt = format_binary_prediction(score, threshold=threshold)

    # Original image for display
    orig_rgb = bytes_to_rgb_u8(raw_bytes)

    # Grad-CAM — never raises; returns zero heatmap + None layer on failure
    heatmap, layer_used = try_generate_gradcam(
        loaded_model.model,
        batch,
        target_layer_name=loaded_model.gradcam_layer,
    )

    heat_rgb    = heatmap_to_rgb_u8(heatmap, orig_rgb.shape[:2])
    overlay_rgb = blend_overlay(orig_rgb, heat_rgb, alpha=0.75)

    original_b64 = array_to_png_base64(orig_rgb)
    heatmap_b64  = array_to_png_base64(heat_rgb)    if layer_used is not None else None
    overlay_b64  = array_to_png_base64(overlay_rgb) if layer_used is not None else None

    return PredictionResponse(
        predicted_label=fmt.predicted_label_display,
        confidence=fmt.confidence,
        probabilities=fmt.probabilities_display,
        heatmap_base64=heatmap_b64,
        overlay_base64=overlay_b64,
        original_base64=original_b64,
        meta=PredictionMeta(
            input_size=[loaded_model.image_size, loaded_model.image_size],
            model_name=f"{loaded_model.backbone} ({model_path_name})",
            gradcam_layer=layer_used,
        ),
        model_summary=loaded_model.summary,
    )
