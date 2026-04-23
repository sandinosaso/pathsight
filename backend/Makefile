.PHONY: docker_up

# Build the backend Docker image and deploy to Cloud Run.
# Requires GCP_PROJECT_ID and GCP_REGION env vars to be set.
GCP_PROJECT_ID ?= $(error GCP_PROJECT_ID is not set)
GCP_REGION     ?= us-central1
IMAGE           = gcr.io/$(GCP_PROJECT_ID)/pathsight-backend
SERVICE         = pathsight-backend

docker_up:
	docker build -t $(IMAGE) .
	docker push $(IMAGE)
	gcloud run deploy $(SERVICE) \
		--image $(IMAGE) \
		--region $(GCP_REGION) \
		--platform managed \
		--allow-unauthenticated \
		--project $(GCP_PROJECT_ID)
