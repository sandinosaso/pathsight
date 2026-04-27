import os

# Must be set before TensorFlow is imported anywhere in this process.
# On Cloud Run (CPU-only) this suppresses the CUDA device probe entirely.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from contextlib import asynccontextmanager
from pathlib import Path
import tensorflow as tf
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from model.src.model_service.preprocess.dataset_builder import _preprocess_image
from model.src.model_service.interpretability.overlays import bytes_to_png_base64
from model.src.model_service.config import ModelServiceConfig
from backend.src.logic.postprocessprediction import format_binary_prediction
from backend.src.logic.predict import load_model_trained, predict_logic
from backend.src.schemas import PredictionMeta, PredictionResponse


config = ModelServiceConfig()

MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model after uvicorn binds the port so Cloud Run's startup
    probe succeeds before the (slow) Keras load begins."""
    global MODEL
    tf.config.set_visible_devices([], "GPU")
    MODEL = load_model_trained()
    yield


app = FastAPI(lifespan=lifespan)

# CORS – safety net; Firebase proxy eliminates cross-origin requests in production
_cors_origins = os.getenv("CORS_ORIGINS", "https://pathsight.web.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api")

EXAMPLES_DIR = Path(__file__).parent / "examples"


@app.get("/")
async def root():
    return {"status": "ok"}


@router.get("/examples")
async def list_examples():
    items = []
    for label, folder in [("cancer", "cancer"), ("no-cancer", "no_cancer")]:
        d = EXAMPLES_DIR / folder
        if not d.exists():
            continue
        for img_path in sorted(d.glob("*.png")):
            stem = img_path.stem
            items.append({
                "id": stem,
                "filename": img_path.name,
                "label": label,
                "description": f"{'Metastatic' if label == 'cancer' else 'Normal'} tissue patch",
                "image_url": f"/api/examples/{stem}/image",
            })
    return {"examples": items}


@router.get("/examples/{example_id}/image")
async def get_example_image(example_id: str):
    # Prevent path traversal: only allow alphanumeric + underscore
    if not example_id.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid example id")
    for folder in ("cancer", "no_cancer"):
        candidate = EXAMPLES_DIR / folder / f"{example_id}.png"
        if candidate.exists():
            return FileResponse(candidate, media_type="image/png")
    raise HTTPException(status_code=404, detail="Example not found")


@router.post("/predict")
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

    # Step 6: Encode original image as PNG base64
    original_b64 = bytes_to_png_base64(contents)

    return PredictionResponse(
        predicted_label="cancer" if cancer_pc > no_cancer_pc else "no-cancer",
        confidence=format_binary_prediction(result_score).confidence,
        probabilities={
            "cancer": cancer_pc / 100,
            "no-cancer": no_cancer_pc / 100,
        },
        heatmap_base64=None,
        overlay_base64=None,
        original_base64=original_b64,
        meta=PredictionMeta(
            input_size=config.data.input_shape,
            model_name=config.data.best_model_path.name,
            gradcam_layer=None,
        ),
    ).to_dict()


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
