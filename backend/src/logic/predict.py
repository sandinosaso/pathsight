import keras
from PIL import Image
import numpy as np
import io
import os
import tensorflow as tf

def load_model_trained():


    current_dir = os.path.dirname(os.path.abspath(__file__))

    #TODO: Update the model path if your model is located elsewhere or has a different name
    model_path = os.path.join(current_dir, "..", "..", "baseline_nb.keras")

    print('✅ Model_loaded')

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
