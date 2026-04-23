import keras
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from model.src.model_service.config import ModelServiceConfig


def load_model_trained():

    config = ModelServiceConfig()

    model_path = config.data.best_model_path

    print('✅ Model_loaded:', model_path)

    # Normalize the path to remove the '..' and make it clean
    model_path = os.path.normpath(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print(f"📦 Loading model from: {model_path}")
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_bytes: bytes):
    # TODO: This is a placeholder. You should implement the actual preprocessing steps based on how your model was trained.

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((96, 96)) # Change this to your model's input size
    img_array = np.array(img) / 255.0 # Normalizing
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

def predict_logic(model, img_data):
    # verbose=0 stops the [1/1] [========] progress bar in the logs
    # TODO: Ensure that the output of the model is a single scalar value. If your model outputs a different shape, you may need to adjust this code accordingly.
    prediction = model.predict(img_data)

    # Extract the single scalar value
    result = float(prediction[0][0])

    print(f"✅ Result predicted: {result:.4f}")

    return result


def predict_logicc(model, img_data: tf.Tensor) -> float:
    # Step 1: Add batch dimension (1, H, W, C)
    img_data = tf.expand_dims(img_data, axis=0)

    # Step 2: Predict — verbose=0 suppresses the progress bar
    prediction = model.predict(img_data, verbose=0)

    # Step 3: Extract single scalar value
    result = float(prediction[0][0])

    print(f"✅ Result predicted: {result:.4f}")

    return result
