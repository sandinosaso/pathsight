# 🧬 Cancer Detection API

A high-performance backend service built with **FastAPI** and **Python 3.10.6** designed to classify medical images and provide confidence scores for cancer detection.

## 🛠 Tech Stack
* **Language:** Python 3.10.6
* **API Framework:** [FastAPI](https://fastapi.tiangolo.com/)
* **Containerization:** Docker

---

## 🏗 Project Structure

```text
.
├── backend/
│   ├── src/
│   │   ├── logic/
│   │   │   ├── postprocessprediction.py  # Formats raw scores into human-readable JSON
│   │   │   └── predict.py                # Handles model loading and inference logic
│   │   ├── main.py                       # FastAPI application entry point
│   │   └── schemas.py                    # Pydantic models (Response/Meta objects)
│   ├── dockerfile                        # Docker configuration
│   ├── Makefile                          # Task automation (install, run, docker)
│   ├── requirements.txt                  # Production dependencies
│   └── README.md                         # Project documentation
```

---

## 🚀 Getting Started

### 1. Environment Configuration
Create a `.env` file in the root directory to define your ports and image names:
```env
DOCKER_REPO_NAME=pathsight
DOCKER_IMAGE_NAME=cancer-detection-api
DOCKER_LOCAL_PORT=8080
APP_PORT=8080
```

### 2. Installation
We use a centralized Makefile to manage dependencies across the core model, notebooks, and backend:
```bash
# Install all dependencies (Backend + Notebooks + Model)
make install-all
```

### 3. Running the API

**Development Mode (Hot Reload):**
This uses the APP_PORT defined in your .env.
```bash
make run-api-dev
```

**Production Mode (Docker):**
Build and start the container in one step:
```bash
make docker-up
```

---

## 🛰 API Documentation

Once the server is running, explore the interactive documentation at:
👉 **Swagger UI:** `http://localhost:[YOUR_PORT]/docs`

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check. Returns `200 OK`. |
| `POST` | `/predict` | Accept image and return detection probabilities. |

### Example Response (`POST /predict`)
```json
{
  "predicted_label": "cancer",
  "confidence": 0.92,
  "probabilities": {
    "cancer": 0.92,
    "no-cancer": 0.08
  },
  "original_base64": "...",
  "meta": {
    "input_size": [224, 224],
    "model_name": "baseline_nb.keras",
    "gradcam_layer": null
  }
}
```
