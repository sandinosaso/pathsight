from fastapi import FastAPI, UploadFile, File
# Ensure your logic folder is accessible in the python path
from backend.src.logic.predict import load_model_trained, predict_logic
from backend.src.schemas import PredictionMeta, PredictionResponse
from model.src.model_service.preprocess.dataset_builder import _preprocess_image
import tensorflow as tf
from model.src.model_service.config import ModelServiceConfig
from backend.src.logic.postprocessprediction import format_binary_prediction


config = ModelServiceConfig()

app = FastAPI()

MODEL = load_model_trained()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    # Step 1: Read the bytes from the uploaded file
    contents = await img.read()

    # Step 2: Decode bytes into a tf.Tensor
    image = tf.image.decode_image(
        tf.constant(contents),
        channels=3,
        expand_animations=False,
    )

    # Step 3: Preprocess using _preprocess_image
    image, _ = _preprocess_image(
        image,
        tf.constant(0),  # dummy label, not needed for inference
        image_size=config.data.image_size,
        augment=False,   # never augment at inference time
    )

    # Step 4: Run inference
    result_score = predict_logic(model=MODEL, img_data=image)

    # Step 5: Calculate percentages
    cancer_pc = result_score * 100
    no_cancer_pc = (1.0 - result_score) * 100

    return PredictionResponse(
        predicted_label="cancer" if cancer_pc > no_cancer_pc else "no-cancer",
        confidence=format_binary_prediction(result_score).confidence,
        probabilities={
            "cancer": cancer_pc / 100,
            "no-cancer": no_cancer_pc / 100,
        },
        meta=PredictionMeta(
            input_size=config.data.input_shape,
            model_name=config.data.best_model_path.name
        ),
    ).to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
