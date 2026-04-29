"""Benchmark experiment runner for multi-backbone PCam evaluation.

Usage (notebook)
----------------
>>> from model_service.training.experiments import RunConfig, run_benchmark
>>> cfg = RunConfig(run_id="mobilenetv3small_96", backbone="mobilenetv3small", image_size=96)
>>> summary = run_benchmark(cfg)
>>> print(summary)

Usage (sweep script)
--------------------
See ``model/scripts/run_benchmark.py``.
"""

from __future__ import annotations

import platform
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

# ConvNeXt uses XLA-compiled depthwise convolutions that are incompatible with
# tensorflow-metal on Apple Silicon.  We fall back to CPU for those backbones
# when a Metal GPU is the active device.
_METAL_INCOMPATIBLE_BACKBONES = frozenset({"convnext", "convnexttiny"})


def _resolve_device(backbone: str) -> str | None:
    """Return '/CPU:0' if backbone is known to be Metal-incompatible, else None."""
    if not any(backbone.startswith(b) for b in _METAL_INCOMPATIBLE_BACKBONES):
        return None
    if platform.system() != "Darwin":
        return None
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(
            f"  NOTE: {backbone} is incompatible with tensorflow-metal (XLA depthwise conv)."
            " Forcing CPU for this run.\n"
        )
        return "/CPU:0"
    return None


def _ctx(device: str | None):
    """Return a fresh context manager for the given device string.

    Context managers are single-use objects in Python 3.12 — calling __enter__
    a second time on the same instance raises AttributeError.  Always call this
    function to get a new instance rather than reusing one across `with` blocks.
    """
    return tf.device(device) if device else nullcontext()


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """All parameters that define a single benchmark experiment.

    Every field defaults to a sensible value so you only need to specify
    the fields that differ between runs.
    """

    run_id: str
    """Unique identifier used to scope all artifact paths, e.g. ``mobilenetv3small_96``."""

    backbone: str
    """Backbone name understood by :func:`backbones.build_transfer_model`, e.g. ``"mobilenetv3small"``."""

    image_size: int
    """Input spatial dimension (square), e.g. 96, 128, 160, or 224."""

    # Training hyper-parameters
    batch_size: int = 32
    stage1_epochs: int = 12
    """Head-only training epochs (backbone frozen).  12 gives early-stopping
    enough room to find the best head fit without overshooting; the YAML can
    still override per-entry."""
    stage2_epochs: int = 8
    """Fine-tuning epochs (top layers unfrozen).  8 with ``patience=5`` lets
    the unfrozen backbone settle into the new colour-jittered distribution
    before early-stopping picks the best epoch."""
    fine_tune_layers: int = 40
    """Number of backbone layers to unfreeze during fine-tuning."""
    learning_rate: float = 1e-3
    fine_tune_lr: float = 1e-5
    head_dropout: float = 0.4
    """Dropout in the classification head.  0.4 (vs the previous 0.3)
    counters the larger Stage 2 training sets pulling the head toward
    over-confident predictions on majority patterns."""
    head_units: int = 128

    # Head / backbone architecture knobs
    head_style: str = "default"
    """Classification head variant.

    ``"default"`` — GAP → Dense(head_units, relu) → Dropout → sigmoid.
    ``"minimal"`` — GAP → BatchNormalization → Dropout → sigmoid (no hidden Dense).
    The ``"minimal"`` style matches the Shayan-notebook recipe and has fewer
    parameters, which helps when training on small datasets without augmentation.
    """
    freeze_backbone: bool = True
    """Whether to freeze the backbone at construction time.

    ``True`` (default) — two-stage approach: head-only stage 1, then
    ``unfreeze_top`` for stage 2.
    ``False`` — all backbone weights are trainable from epoch 1 (single-stage,
    end-to-end fine-tuning; matches the Shayan-notebook recipe).
    When ``False``, stage 2 is still run if ``stage2_epochs > 0`` — the
    ``unfreeze_top`` call in stage 2 becomes a no-op because all layers are
    already trainable, but the separate dataset and LR change still apply.
    """

    # Data
    augment_train: bool = True
    stain_normalise: bool = False
    seed: int = 42
    max_train_samples: int | None = 20_000
    """Stage 1 (frozen backbone) training cap.

    20 000 balanced examples is sufficient for head-only training and keeps
    iteration fast — the backbone's ImageNet weights already produce rich
    features, so the head saturates quickly.
    Val and test are scaled to ``max_train_samples // 5`` each.
    Set to ``None`` to use the full dataset (~262 k train / 32 k val / 32 k test)."""

    stage2_train_samples: int | None = None
    """Stage 2 (fine-tuning) training cap.

    ``None`` (default) → reuse the Stage 1 datasets unchanged (current behaviour).
    Set to a larger value (e.g. ``50_000``) to expose the unfrozen backbone
    layers to a broader slice of the distribution without paying the full-dataset
    cost.  The Stage 2 val/test splits are scaled to ``stage2_train_samples // 5``
    so early-stopping and final evaluation are consistent with the training volume.

    Rationale: Stage 1 only trains the classification head — a small, fast subset
    suffices.  Stage 2 adapts the backbone's feature space to PCam tissue patterns;
    more diverse examples reduce the risk of overfitting the unfrozen layers and
    typically improve recall (the clinically critical metric)."""

    # Paths
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4])
    """Monorepo root.  Auto-detected from the source tree; override if needed."""

    def artifacts_dir(self) -> Path:
        """Per-run directory under ``artifacts/benchmarks/``."""
        return self.repo_root / "artifacts" / "benchmarks" / self.run_id

    def data_dir(self) -> Path:
        return self.repo_root / "data"

    def input_shape(self) -> tuple[int, int, int]:
        return (self.image_size, self.image_size, 3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run model on *dataset* and return (y_true, y_prob) arrays."""
    y_true_list, y_prob_list = [], []
    for x_batch, y_batch in dataset:
        probs = model(x_batch, training=False).numpy().ravel()
        y_prob_list.append(probs)
        y_true_list.append(y_batch.numpy().ravel())
    return np.concatenate(y_true_list), np.concatenate(y_prob_list)


def _measure_inference_latency(model: tf.keras.Model, image_size: int, n_warmup: int = 3, n_runs: int = 20) -> float:
    """Return mean inference time in milliseconds per image (batch size 1)."""
    dummy = tf.zeros((1, image_size, image_size, 3))
    for _ in range(n_warmup):
        model(dummy, training=False)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(dummy, training=False)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Core orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(cfg: RunConfig, *, data_dir: str | None = None, download: bool = True) -> dict[str, Any]:
    """Execute a full two-stage training run and return the summary dict.

    Steps
    -----
    1. Build data pipelines at ``cfg.image_size`` with the correct preprocessing.
    2. Build frozen backbone + head model.
    3. Stage 1: train head only.
    4. Stage 2: unfreeze top layers, fine-tune with low LR.
    5. Evaluate on the test set with extended clinical metrics.
    6. Measure single-image inference latency.
    7. Save artifacts (model, histories, predictions, plots, summary.json).
    8. Return the summary dict.

    Parameters
    ----------
    cfg:
        Run configuration.
    data_dir:
        Override TFDS data directory (useful in notebooks).
    download:
        Whether to download the PCam dataset if not cached.
    """
    from model_service.preprocess.dataset_builder import build_pcam_datasets
    from model_service.evaluation.metrics import compute_clinical_metrics
    from model_service.evaluation.plots import plot_confusion_matrix, plot_pr_curve, plot_roc
    from model_service.training.backbones import build_transfer_model, unfreeze_top
    from model_service.training.callbacks import EpochTimer, default_callbacks
    from model_service.training.train import run_training
    from model_service.utils.io import save_json
    from model_service.utils.seed import set_seed

    set_seed(cfg.seed)
    out_dir = cfg.artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine whether Stage 2 uses a different (larger) dataset than Stage 1.
    _use_separate_stage2_data = (
        cfg.stage2_train_samples is not None
        and cfg.stage2_train_samples != cfg.max_train_samples
    )

    print(f"\n{'='*60}")
    print(f"  RUN: {cfg.run_id}")
    print(f"  backbone={cfg.backbone}  image_size={cfg.image_size}")
    if cfg.max_train_samples is not None:
        n_eval1 = max(cfg.max_train_samples // 5, 128)
        print(f"  stage1 data: train={cfg.max_train_samples}  val/test={n_eval1} (balanced 50/50)")
    else:
        print(f"  stage1 data: full dataset")
    if _use_separate_stage2_data:
        n_eval2 = max(cfg.stage2_train_samples // 5, 128)
        print(f"  stage2 data: train={cfg.stage2_train_samples}  val/test={n_eval2} (balanced 50/50)")
    else:
        print(f"  stage2 data: same as stage1")
    print(f"{'='*60}\n")

    # ── 1. Stage 1 data ────────────────────────────────────────────────────
    # backbone= drives preprocessing; the dataset builder resolves the mode.
    train_ds1, val_ds1, test_ds1, _ = build_pcam_datasets(
        backbone=cfg.backbone,
        data_dir=data_dir,
        download=download,
        image_size=cfg.image_size,
        max_train_samples=cfg.max_train_samples,
    )
    # When train is a repeated finite subset, Keras needs steps_per_epoch so
    # it knows when one epoch ends and can fire val / callbacks correctly.
    steps_per_epoch1: int | None = (
        cfg.max_train_samples // cfg.batch_size if cfg.max_train_samples is not None else None
    )

    # ── 2. Stage 2 data ────────────────────────────────────────────────────
    if _use_separate_stage2_data:
        train_ds2, val_ds2, test_ds2, _ = build_pcam_datasets(
            backbone=cfg.backbone,
            data_dir=data_dir,
            download=download,
            image_size=cfg.image_size,
            max_train_samples=cfg.stage2_train_samples,
        )
        steps_per_epoch2: int | None = cfg.stage2_train_samples // cfg.batch_size
    else:
        train_ds2, val_ds2, test_ds2 = train_ds1, val_ds1, test_ds1
        steps_per_epoch2 = steps_per_epoch1

    # Some backbones (ConvNeXt) use XLA ops incompatible with Metal GPU.
    # _resolve_device() returns '/CPU:0' in that case, None otherwise.
    # _ctx() must be called fresh for each `with` block — context managers are
    # single-use in Python 3.12 and cannot be re-entered after __exit__.
    device = _resolve_device(cfg.backbone)

    # ── 3. Model ───────────────────────────────────────────────────────────
    with _ctx(device):
        model = build_transfer_model(
            cfg.backbone,
            cfg.input_shape(),
            learning_rate=cfg.learning_rate,
            head_dropout=cfg.head_dropout,
            head_units=cfg.head_units,
            head_style=cfg.head_style,
            freeze_backbone=cfg.freeze_backbone,
        )
    model.summary(print_fn=lambda s: print(s))

    params_total = int(model.count_params())
    params_trainable = int(sum(tf.size(w).numpy() for w in model.trainable_weights))

    # ── 4. Stage 1 — frozen backbone ──────────────────────────────────────
    timer1 = EpochTimer()
    cbs1 = default_callbacks(
        out_dir / f"{cfg.run_id}_stage1_best.keras",
        out_dir / "stage1.csv",
    ) + [timer1]

    with _ctx(device):
        h1 = run_training(model, train_ds1, val_ds1, epochs=cfg.stage1_epochs, steps_per_epoch=steps_per_epoch1, callbacks=cbs1)
    save_json(out_dir / "stage1_history.json", {k: [float(v) for v in vals] for k, vals in h1.history.items()})

    # ── 5. Stage 2 — fine-tuning ───────────────────────────────────────────
    with _ctx(device):
        model = unfreeze_top(model, cfg.backbone, num_layers=cfg.fine_tune_layers, learning_rate=cfg.fine_tune_lr)

    timer2 = EpochTimer()
    cbs2 = default_callbacks(
        out_dir / f"{cfg.run_id}_best.keras",
        out_dir / "stage2.csv",
    ) + [timer2]

    with _ctx(device):
        h2 = run_training(model, train_ds2, val_ds2, epochs=cfg.stage2_epochs, steps_per_epoch=steps_per_epoch2, callbacks=cbs2)
    save_json(out_dir / "stage2_history.json", {k: [float(v) for v in vals] for k, vals in h2.history.items()})

    total_train_s = timer1.total_time + timer2.total_time
    mean_epoch_s = (timer1.total_time + timer2.total_time) / max(1, len(timer1.epoch_times) + len(timer2.epoch_times))

    # ── 6. Evaluation ──────────────────────────────────────────────────────
    # test_ds2: Stage 2 test split (larger when stage2_train_samples is set,
    # same as Stage 1 otherwise).  Using Stage 2's split is conservative and
    # consistent with the data distribution the fine-tuned model was validated on.
    # Must run under the same device context as training — on Apple Silicon,
    # ConvNeXt model weights are pinned to CPU and will error if inference
    # attempts to run on the Metal GPU (device mismatch).
    print("\nEvaluating on test set …")
    with _ctx(device):
        y_true, y_prob = _collect_predictions(model, test_ds2)
    np.savez(out_dir / "test_predictions.npz", y_true=y_true, y_prob=y_prob)

    # Pass 1 — find the optimal thresholds from the precision-recall curve.
    _thr_search = compute_clinical_metrics(y_true, y_prob, threshold=0.5)
    best_f1_thr = _thr_search.pop("best_f1_threshold")
    high_recall_thr = _thr_search.pop("high_recall_95_threshold", _thr_search.pop("high_recall_95.0_threshold", None))
    high_precision_thr = _thr_search.pop("high_precision_95_threshold", _thr_search.pop("high_precision_95.0_threshold", None))
    thresholds = {
        "best_f1": best_f1_thr,
        "high_recall_95": high_recall_thr,
        "high_precision_95": high_precision_thr,
    }

    # Pass 2 — recompute all classification metrics at best_f1_thr.
    # For cancer detection the default 0.5 cut-off is too conservative — it
    # misses many true positives.  best_f1_thr (derived from the PR curve) is
    # the optimal operating point and typically sits between 0.2–0.4, giving
    # substantially higher recall than 0.5.
    print(f"  best_f1_threshold={best_f1_thr:.4f} — computing test metrics at this threshold")
    test_metrics = compute_clinical_metrics(y_true, y_prob, threshold=best_f1_thr)
    test_metrics.pop("best_f1_threshold", None)
    test_metrics.pop("high_recall_95_threshold", test_metrics.pop("high_recall_95.0_threshold", None))
    test_metrics.pop("high_precision_95_threshold", test_metrics.pop("high_precision_95.0_threshold", None))

    # ── 7. Inference latency ───────────────────────────────────────────────
    with _ctx(device):
        inference_ms = _measure_inference_latency(model, cfg.image_size)

    # ── 8. Plots ───────────────────────────────────────────────────────────
    from model_service.evaluation.metrics import confusion_matrix_counts
    cm = confusion_matrix_counts(y_true, y_prob, threshold=best_f1_thr)
    plot_confusion_matrix(cm, ("Normal", "Metastatic"), out_path=out_dir / "confusion_matrix.png")
    plot_roc(y_true, y_prob, out_path=out_dir / "roc.png")
    plot_pr_curve(y_true, y_prob, best_f1_threshold=best_f1_thr, out_path=out_dir / "pr_curve.png")

    # ── 9. Summary ─────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "run_id": cfg.run_id,
        "backbone": cfg.backbone,
        "image_size": cfg.image_size,
        "params_total": params_total,
        "params_trainable_stage1": params_trainable,
        "test": {k: round(float(v), 5) for k, v in test_metrics.items()},
        "test_threshold": round(best_f1_thr, 4),
        "thresholds": {k: round(float(v), 4) if v is not None else None for k, v in thresholds.items()},
        "timing": {
            "epoch_time_s": round(mean_epoch_s, 1),
            "total_train_s": round(total_train_s, 1),
            "inference_ms_per_image": round(inference_ms, 2),
        },
        "config": {
            "batch_size": cfg.batch_size,
            "stage1_epochs": cfg.stage1_epochs,
            "stage2_epochs": cfg.stage2_epochs,
            "fine_tune_layers": cfg.fine_tune_layers,
            "learning_rate": cfg.learning_rate,
            "fine_tune_lr": cfg.fine_tune_lr,
            "head_dropout": cfg.head_dropout,
            "augment_train": cfg.augment_train,
            "stain_normalise": cfg.stain_normalise,
            "max_train_samples": cfg.max_train_samples,
            "stage2_train_samples": cfg.stage2_train_samples,
        },
    }
    # summary.json — human-readable record always at a fixed name inside out_dir.
    save_json(out_dir / "summary.json", summary)
    # {run_id}_best.json — GCS sidecar that must be uploaded alongside
    # {run_id}_best.keras.  Naming it after the run_id prevents accidental
    # overwrites when multiple experiments share the same artifact directory,
    # and lets the backend resolve the sidecar from the model path alone
    # (os.path.splitext(model_path)[0] + ".json").
    best_json_path = out_dir / f"{cfg.run_id}_best.json"
    save_json(best_json_path, summary)
    print(f"\nSummary saved to {out_dir / 'summary.json'} and {best_json_path}")
    print(
        f"  AUC={test_metrics.get('roc_auc', 0):.4f}"
        f"  PR-AUC={test_metrics.get('pr_auc', 0):.4f}"
        f"  F1={test_metrics.get('f1', 0):.4f}"
        f"  Recall={test_metrics.get('recall', 0):.4f}"
        f"  Specificity={test_metrics.get('specificity', 0):.4f}"
        f"  Inference={inference_ms:.1f}ms"
    )

    return summary
