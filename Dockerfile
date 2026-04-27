# Use an official Python runtime as a parent image
FROM python:3.10.6-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install Python deps. build-essential is required only during pip's source
# builds; purge it afterwards so it doesn't bloat the final image layer.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y build-essential \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code into the container
COPY backend /app/backend
COPY model /app/model

RUN pip install -e model

# The model file is fetched from GCS by the CI pipeline (or locally via
# `python backend/scripts/download_model.py`) before this image is built,
# so it is part of the build context and we just COPY it in. We do not
# download from GCS during `docker build` because the docker build sandbox
# has no Application Default Credentials.
COPY artifacts/models/best_model.keras /app/artifacts/models/best_model.keras
ENV BEST_MODEL_PATH=/app/artifacts/models/best_model.keras

# Set PYTHONPATH to /app so it can see the "backend" folder
ENV PYTHONPATH=/app

EXPOSE 8080
# Run the application — Cloud Run injects PORT at runtime
CMD ["sh", "-c", "uvicorn backend.src.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 2"]
