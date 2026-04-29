"""Keras training callbacks."""

from __future__ import annotations

import time
from pathlib import Path

from tensorflow import keras

from model_service.config import ModelServiceConfig


class EpochTimer(keras.callbacks.Callback):
    """Record wall-clock time for each epoch.

    After training, ``epoch_times`` contains one float per completed epoch
    (seconds), and ``total_time`` gives the sum.

    Example
    -------
    >>> timer = EpochTimer()
    >>> model.fit(..., callbacks=[timer])
    >>> print(timer.epoch_times)   # [142.3, 138.9, ...]
    >>> print(timer.total_time)    # 281.2
    """

    def __init__(self) -> None:
        super().__init__()
        self.epoch_times: list[float] = []
        self._start: float = 0.0

    @property
    def total_time(self) -> float:
        return sum(self.epoch_times)

    @property
    def mean_epoch_time(self) -> float:
        return self.total_time / len(self.epoch_times) if self.epoch_times else 0.0

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        self.epoch_times.append(time.perf_counter() - self._start)


def default_callbacks(
    checkpoint_path: Path | None,
    csv_log_path: Path | None,
    *,
    early_stopping_patience: int | None = None,
    monitor: str | None = None,
    mode: str | None = None,
) -> list[keras.callbacks.Callback]:
    """Return the default callback stack for a training run.

    Callbacks included:
    - ``ReduceLROnPlateau`` — fires *before* early stopping (shorter patience)
      to give the optimizer one last chance at a lower step size.
    - ``EarlyStopping`` — terminates training when the monitor has not improved
      for ``patience`` epochs; restores best weights automatically.
    - ``ModelCheckpoint`` — saves the best checkpoint to ``checkpoint_path``.
    - ``CSVLogger`` — appends per-epoch metrics to ``csv_log_path``.

    All tunable knobs default to :class:`ModelServiceConfig` values (driven by
    env vars) so experiments can change them without touching source code.
    """
    cfg = ModelServiceConfig().train
    if monitor is None:
        monitor = cfg.early_stop_monitor
    if mode is None:
        mode = cfg.early_stop_mode
    if early_stopping_patience is None:
        early_stopping_patience = cfg.early_stopping_patience

    # ReduceLROnPlateau fires first (patience < early_stopping_patience) so the
    # optimizer gets at least one LR reduction before training is terminated.
    # Using the same monitor as EarlyStopping keeps both callbacks responding to
    # the same clinical signal (val_pr_auc by default).
    cbs: list[keras.callbacks.Callback] = [
        keras.callbacks.ReduceLROnPlateau(
            monitor=cfg.reduce_lr_monitor,
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.reduce_lr_min_lr,
            cooldown=cfg.reduce_lr_cooldown,
            mode=mode,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode=mode,
        ),
    ]
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        cbs.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=monitor,
                save_best_only=True,
                mode=mode,
            )
        )
    if csv_log_path is not None:
        csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        cbs.append(keras.callbacks.CSVLogger(str(csv_log_path)))
    return cbs
