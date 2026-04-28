"""Generic training loop."""

from __future__ import annotations

from typing import Any

from tensorflow import keras


def run_training(
    model: keras.Model,
    train_ds: Any,
    val_ds: Any,
    *,
    epochs: int,
    steps_per_epoch: int | None = None,
    callbacks: list[keras.callbacks.Callback] | None = None,
) -> keras.callbacks.History:
    """Fit model on tf.data datasets.

    Parameters
    ----------
    steps_per_epoch:
        Required when ``train_ds`` is infinite (i.e. built with ``.repeat()``).
        Set to ``max_train_samples // batch_size`` so Keras knows when one
        epoch ends and EarlyStopping / checkpointing fire correctly.
        Leave as ``None`` for finite datasets (full PCam split).
    """
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks or [],
        verbose=2,
    )
