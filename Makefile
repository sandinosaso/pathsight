.PHONY: install install-notebooks install-all test frontend-install frontend-dev frontend-build

api:
	pip install fastapi uvicorn python-multipart python-dotenv
run:
	#TODO: Use port from env.
	uvicorn backend.src.main:app --reload --port 8080

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

## Install frontend dependencies
frontend-install:
	npm ci --prefix frontend

## Start the frontend Vite dev server (http://localhost:5173)
frontend-dev: frontend-install
	npm run dev --prefix frontend

## Build the frontend for production (output: frontend/dist/)
frontend-build: frontend-install
	npm run build --prefix frontend
