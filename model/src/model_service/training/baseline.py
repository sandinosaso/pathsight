import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_service.config import ModelServiceConfig

MONITOR = 'val_loss' # val_auc
MODE = 'min' # max

config = ModelServiceConfig()

def default_callbacks():
    stopper = EarlyStopping(
        monitor=MONITOR,
        patience=3,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath=f"{config.paths.artifacts_checkpoints}/baseline_cnn_checkpoint.keras",
        monitor=MONITOR,
        save_best_only=True,
        mode=MODE
    )

    return [stopper, checkpoint]

def build_baseline_cnn(input_shape=config.data.input_shape, learning_rate=config.train.learning_rate):
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
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def run_training(model: tf.keras.Model, train_ds, val_ds, epochs=config.train.epochs, steps_per_epoch=None, validation_steps_per_epoch=None, callbacks=default_callbacks()):

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps_per_epoch,
        callbacks=callbacks
    )
    return history
