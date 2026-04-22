from fastapi import FastAPI, UploadFile, File
# Ensure your logic folder is accessible in the python path
from backend.src.logic.predict import load_model_trained, preprocess_image, predict_logic

app = FastAPI()

MODEL = load_model_trained()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    # Read the bytes from the uploaded file
    contents = await img.read()

    # 3. Preprocess the image bytes
    input_data = preprocess_image(contents)

    # 4. Run inference
    result_score = predict_logic(model=MODEL, img_data=input_data)

    # 5. Calculate percentages
    cancer_pc = result_score * 100
    no_cancer_pc = (1.0 - result_score) * 100

    return {
        "prediction": {
            "cancer": f"{cancer_pc:02.0f}%",
            "no-cancer": f"{no_cancer_pc:02.0f}%"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
