from dataclasses import dataclass, asdict
from typing import Optional

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

    def to_dict(self):
        return asdict(self)
