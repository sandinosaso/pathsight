from dataclasses import dataclass, asdict, field
from typing import Any, Optional

@dataclass
class PredictionMeta:
    input_size: list[int]
    model_name: str
    gradcam_layer: Optional[str] = None

@dataclass
class PredictionResponse:
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    heatmap_base64: Optional[str]
    overlay_base64: Optional[str]
    original_base64: str
    meta: PredictionMeta
    model_summary: Optional[dict[str, Any]] = field(default=None)
    """Full best.json sidecar for the loaded model.

    Contains backbone, image_size, test metrics (at the deployment threshold),
    thresholds, timing, and training config — everything the frontend needs to
    show model quality information alongside a prediction.
    """

    def to_dict(self):
        return asdict(self)
