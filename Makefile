# 1. Load environment variables from .env
# This allows the Makefile to use DOCKER_IMAGE_NAME and DOCKER_LOCAL_PORT
ifneq (,$(wildcard ./.env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

.PHONY: install install-notebooks install-all test test-smoke api run docker-build-local docker-run-local docker-up frontend-install frontend-dev frontend-build upload-model

ifneq (,$(wildcard .env))
  include .env
  export
endif

APP_PORT ?= 8080

## Install core model dependencies (clean-syncs, removes unlisted packages)
install:
	pip install --quiet pip-tools
	pip cache purge
	pip-sync requirements.txt

## Install notebook-only dependencies on top of the core ones
install-notebooks: install
	pip-sync requirements.txt notebooks/requirements.txt

## Install everything (core + notebooks)
install-all:
	pip install --quiet pip-tools
	pip cache purge
	pip-sync requirements.txt notebooks/requirements.txt backend/requirements.txt
	pip install -r requirements.txt
	pip install -r backend/requirements.txt
	pip install -r notebooks/requirements.txt
	pip install -e model

## Install frontend dependencies
frontend-install:
	npm ci --prefix frontend

## Run the test suite
test:
	pytest

## Run model smoke tests only
test-smoke:
	pytest backend/tests/ -v

run-api-dev:
	uvicorn backend.src.main:app --reload --reload-dir backend --port $(APP_PORT)

## Start the frontend Vite dev server (http://localhost:5173)
run-frontend-dev: frontend-install
	npm run dev --prefix frontend

## Build the frontend for production (output: frontend/dist/)
frontend-build: frontend-install
	npm run build --prefix frontend

## Build the local Docker image
# We use the root context (.) so Docker can see the /artifacts folder
docker-build-local:
	docker build --tag=$(DOCKER_IMAGE_NAME):local -f Dockerfile .

## Run the local Docker container
# Maps your .env port to the internal $PORT and passes the .env file in
docker-run-local:
	docker run \
		-e PORT=$(DOCKER_LOCAL_PORT) \
		-p $(DOCKER_LOCAL_PORT):$(DOCKER_LOCAL_PORT) \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local

## Build and Run in one single command
docker-up: docker-build-local docker-run-local

# ─── GCS model upload ────────────────────────────────────────────────────────
# Upload a trained model (and its JSON summary) to the GCS bucket, overwriting
# whatever is there.  The bucket always stores the files as best_model.keras /
# best_model.json so the CI/CD deploy job picks them up automatically.
#
# Usage (MODEL_PATH is required):
#   make upload-model MODEL_PATH=artifacts/benchmarks/efficientnetb0_224/best_model.keras
#
# The matching .json is derived automatically (same dir, same stem).
# MODEL_BUCKET_NAME is read from .env or can be overridden on the command line.

MODEL_BUCKET_NAME ?= $(error MODEL_BUCKET_NAME is not set — add it to .env or pass it on the command line)

.PHONY: upload-model
upload-model:
ifndef MODEL_PATH
	$(error MODEL_PATH is required — e.g. make upload-model MODEL_PATH=artifacts/benchmarks/efficientnetb0_224/best_model.keras)
endif
	@MODEL_JSON="$$(dirname $(MODEL_PATH))/$$(basename $(MODEL_PATH) .keras).json"; \
	if [ ! -f "$(MODEL_PATH)" ]; then echo "ERROR: model file not found: $(MODEL_PATH)"; exit 1; fi; \
	if [ ! -f "$$MODEL_JSON" ]; then echo "ERROR: summary JSON not found: $$MODEL_JSON"; exit 1; fi; \
	echo "Uploading model  : $(MODEL_PATH)"; \
	gcloud storage cp "$(MODEL_PATH)" "gs://$(MODEL_BUCKET_NAME)/best_model.keras"; \
	echo "Uploading summary: $$MODEL_JSON"; \
	gcloud storage cp "$$MODEL_JSON" "gs://$(MODEL_BUCKET_NAME)/best_model.json"; \
	echo "Done — gs://$(MODEL_BUCKET_NAME)/best_model.keras and best_model.json updated."
