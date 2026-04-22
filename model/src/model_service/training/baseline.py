import json
import datetime
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_service.config import ModelServiceConfig



def default_callbacks(timestamp: str):
    """Creates callbacks using config and a unique timestamp for filenames."""
    config = ModelServiceConfig()
    stopper = EarlyStopping(
        monitor=config.train.early_stop_monitor,
        patience=config.train.early_stopping_patience,
        mode=config.train.early_stop_mode,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath=config.paths.artifacts_checkpoints / f"baseline_{timestamp}.keras",
        monitor=config.train.early_stop_monitor,
        mode=config.train.early_stop_mode,
        save_best_only=True,
        save_weights_only=False,
    )

    return [stopper, checkpoint]

def build_baseline_cnn(input_shape=None, learning_rate=None):
    """Builds model using central config for shape, rate, and metrics."""

    config = ModelServiceConfig()
    if input_shape is None:
        input_shape = config.data.input_shape
    if learning_rate is None:
        learning_rate = config.train.learning_rate

    inputs = layers.Input(shape=input_shape)
    # Layer 1
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Layer 2: Added for better accuracy
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Dropout: The 'Overfitting Guard'
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name="baseline_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=config.train.metrics  # Pulls the list: [acc, auc, precision, recall]
    )
    return model

def run_training(model: tf.keras.Model, train_ds, val_ds):
    """Runs training, calculates F1, and saves metrics to JSON."""
    config = ModelServiceConfig()

    # 1. Create unique timestamp for this specific run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 2. Execute training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.train.epochs,
        callbacks=default_callbacks(config, timestamp)
    )

    return history

def calculate_save_metrics(history):

    config = ModelServiceConfig()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # 1. Extract best metrics (at the point where EarlyStopping restored weights)
    monitor = config.train.early_stop_monitor
    if config.train.early_stop_mode == 'max':
        best_idx = history.history[monitor].index(max(history.history[monitor]))
    else:
        best_idx = history.history[monitor].index(min(history.history[monitor]))

    final_metrics = {m: float(v[best_idx]) for m, v in history.history.items()}

    # 2. Calculate F1-Score manually
    p = final_metrics.get('val_precision', 0)
    r = final_metrics.get('val_recall', 0)
    final_metrics['val_f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    final_metrics['run_timestamp'] = timestamp

    # 3. Save to artifacts/metrics/
    json_path = config.paths.artifacts_metrics / f"metrics_{timestamp}.json"
    config.paths.artifacts_metrics.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"✅ Training Complete. Metrics saved to {json_path}")

    return final_metrics
