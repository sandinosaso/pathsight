# 1. Load environment variables from .env
# This allows the Makefile to use DOCKER_IMAGE_NAME and DOCKER_LOCAL_PORT
ifneq (,$(wildcard ./.env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

.PHONY: install install-notebooks install-all test api run docker_build_local docker_run_local docker_up

api:
	pip install fastapi uvicorn python-multipart python-dotenv
run:
	#TODO: Use port from env.
	uvicorn backend.src.main:app --reload --port $(DOCKER_LOCAL_PORT)

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

## Run the test suite
test:
	pytest

## Build the local Docker image
# We use the root context (.) so Docker can see the /artifacts folder
docker_build_local:
	docker build --tag=$(DOCKER_IMAGE_NAME):local -f backend/Dockerfile .

## Run the local Docker container
# Maps your .env port to the internal $PORT and passes the .env file in
docker_run_local:
	docker run \
		-e PORT=$(DOCKER_LOCAL_PORT) \
		-p $(DOCKER_LOCAL_PORT):$(DOCKER_LOCAL_PORT) \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local

## Build and Run in one single command
docker_up: docker_build_local docker_run_local
