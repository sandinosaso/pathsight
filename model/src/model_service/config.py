"""Central training / data configuration.

All settings can be overridden via environment variables or a ``.env`` file
located in ``model/`` (or the current working directory).  ``python-dotenv``
is loaded once at import time so callers never need to do it themselves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import ast

from dotenv import load_dotenv

# Load .env from model/ (one level above src/) or from cwd, whichever exists.
# Existing env vars take precedence (override=False is the default).
_model_dir = Path(__file__).resolve().parents[2]  # model/src -> model/
load_dotenv(_model_dir / ".env")
load_dotenv()  # also try cwd/.env as a fallback


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    return _env(key, "1" if default else "0") == "1"


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))

def _env_list(key: str, default: list) -> list:
    return ast.literal_eval(_env(key, str(default)))


def _repo_root() -> Path:
    """Monorepo root (``pathsight/``), or override with ``PATHSIGHT_ROOT``."""
    env = os.environ.get("PATHSIGHT_ROOT")
    if env:
        return Path(env).resolve()
    # model/src/model_service/config.py -> parents[3] == pathsight/
    return Path(__file__).resolve().parents[3]


@dataclass
class DataConfig:
    image_size: int = field(default_factory=lambda: _env_int("PCAM_IMAGE_SIZE", 96))
    input_shape: list = field(default_factory=lambda: _env_list("PCAM_INPUT_SHAPE", "[96,96,3]"))
    batch_size: int = field(default_factory=lambda: _env_int("PCAM_BATCH_SIZE", 32))
    seed: int = field(default_factory=lambda: _env_int("PCAM_SEED", 42))
    shuffle_buffer: int = 4096
    cache: bool = True  # applies to val/test only (train set is too large to cache in RAM)
    augment_train: bool = field(default_factory=lambda: _env_bool("PCAM_AUGMENT_TRAIN", True))
    stain_normalise: bool = field(default_factory=lambda: _env_bool("PCAM_STAIN_NORMALISE", False))


@dataclass
class PathsConfig:
    repo_root: Path = field(default_factory=_repo_root)
    artifacts_models: Path | None = None
    artifacts_metrics: Path | None = None
    artifacts_figures: Path | None = None
    artifacts_predictions: Path | None = None
    data_dir: Path | None = None

    def __post_init__(self) -> None:
        root = self.repo_root
        if self.artifacts_models is None:
            self.artifacts_models = root / "artifacts" / "models"
        if self.artifacts_metrics is None:
            self.artifacts_metrics = root / "artifacts" / "metrics"
        if self.artifacts_figures is None:
            self.artifacts_figures = root / "artifacts" / "figures"
        if self.artifacts_predictions is None:
            self.artifacts_predictions = root / "artifacts" / "predictions"
        if self.data_dir is None:
            self.data_dir = root / "data"


@dataclass
class TrainConfig:
    epochs: int = field(default_factory=lambda: _env_int("PCAM_EPOCHS", 10))
    learning_rate: float = field(default_factory=lambda: _env_float("PCAM_LEARNING_RATE", 1e-3))
    fine_tune_lr: float = field(default_factory=lambda: _env_float("PCAM_FINE_TUNE_LR", 1e-5))
    fine_tune_epochs: int = field(default_factory=lambda: _env_int("PCAM_FINE_TUNE_EPOCHS", 5))
    early_stopping_patience: int = field(
        default_factory=lambda: _env_int("PCAM_EARLY_STOPPING_PATIENCE", 3)
    )


@dataclass
class ModelServiceConfig:
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model_name: str = "efficientnetb0_transfer_v1"
    gradcam_layer_name: str | None = None  # auto-detect if None
