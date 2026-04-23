import os
from fastapi import FastAPI, APIRouter, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
# Ensure your logic folder is accessible in the python path
from backend.src.logic.predict import load_model_trained, preprocess_image, predict_logic

app = FastAPI()

# CORS – safety net; Firebase proxy eliminates cross-origin requests in production
_cors_origins = os.getenv("CORS_ORIGINS", "https://pathsight.web.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = load_model_trained()

router = APIRouter(prefix="/api")


@app.get("/")
async def root():
    return {"status": "ok"}


@router.post("/predict")
async def predict(img: UploadFile = File(...)):
    # Read the bytes from the uploaded file
    contents = await img.read()

    # Preprocess the image bytes
    input_data = preprocess_image(contents)

    # Run inference
    result_score = predict_logic(model=MODEL, img_data=input_data)

    # Calculate percentages
    cancer_pc = result_score * 100
    no_cancer_pc = (1.0 - result_score) * 100

    return {
        "prediction": {
            "cancer": f"{cancer_pc:02.0f}%",
            "no-cancer": f"{no_cancer_pc:02.0f}%"
        }
    }


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
