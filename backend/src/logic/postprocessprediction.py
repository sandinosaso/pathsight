"""Format model outputs for API consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_service.constants import LABEL_ALIASES, LABEL_NAMES


@dataclass
class FormattedPrediction:
    predicted_label_internal: str
    predicted_label_display: str
    confidence: float
    probabilities_internal: dict[str, float]
    probabilities_display: dict[str, float]


def format_binary_prediction(prob_positive: float) -> FormattedPrediction:
    """prob_positive is P(metastatic / class 1)."""
    p1 = float(np.clip(prob_positive, 1e-7, 1 - 1e-7))
    p0 = 1.0 - p1
    if p1 >= 0.5:
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
