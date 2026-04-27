# 1. Load environment variables from .env
# This allows the Makefile to use DOCKER_IMAGE_NAME and DOCKER_LOCAL_PORT
ifneq (,$(wildcard ./.env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

.PHONY: install install-notebooks install-all test api run docker-build-local docker-run-local docker-up frontend-install frontend-dev frontend-build

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
	pip-sync requirements.txt notebooks/requirements.txt
	pip install -r requirements.txt
	pip install -r notebooks/requirements.txt
	pip install -e model

## Install frontend dependencies
frontend-install:
	npm ci --prefix frontend

## Run the test suite
test:
	pytest

run-api-dev:
	uvicorn backend.src.main:app --reload --reload-dir backend --port $(APP_PORT)

## Start the frontend Vite dev server (http://localhost:5173)
frontend-dev: frontend-install
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
