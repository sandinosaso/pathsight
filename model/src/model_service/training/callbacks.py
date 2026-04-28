"""Keras training callbacks."""

from __future__ import annotations

import time
from pathlib import Path

from tensorflow import keras


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
    early_stopping_patience: int = 3,
    monitor: str = "val_auc",
    mode: str = "max",
) -> list[keras.callbacks.Callback]:
    cbs: list[keras.callbacks.Callback] = [
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
