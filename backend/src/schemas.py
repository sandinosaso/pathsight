from dataclasses import dataclass, asdict

@dataclass
class PredictionMeta:
    input_size: list[int]
    model_name: str

@dataclass
class PredictionResponse:
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    meta: PredictionMeta

    def to_dict(self):
        return asdict(self)
