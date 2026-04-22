.PHONY: install install-notebooks install-all test

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
