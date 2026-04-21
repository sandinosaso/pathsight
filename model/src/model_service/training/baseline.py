import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
def build_baseline_cnn(input_shape=(224, 224, 3), learning_rate=0.001):
    """
    Builds a baseline CNN for binary tissue classification (Cancer/No Cancer).
    """
    model = models.Sequential([
        # Layer 1: Detecting basic edges and textures
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Layer 2: Detecting more complex shapes
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Layer 3: High-level feature extraction
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten and Dense layers for classification
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Prevents overfitting on specific slides
        layers.Dense(1, activation='sigmoid') # Binary output: 0 (Normal) or 1 (Cancer)
    ])

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model
