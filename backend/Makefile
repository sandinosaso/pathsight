.PHONY: docker_up

# Build the backend Docker image and deploy to Cloud Run.
# Requires GCP_PROJECT_ID and GCP_REGION env vars to be set.
GCP_PROJECT_ID ?= $(error GCP_PROJECT_ID is not set)
GCP_REGION     ?= europe-central2
IMAGE           = $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/pathsight/pathsight-backend
SERVICE         = pathsight-backend

docker_up:
	docker build -t $(IMAGE) -f $(CURDIR)/dockerfile $(CURDIR)/..
	docker push $(IMAGE)
	gcloud run deploy $(SERVICE) \
		--image $(IMAGE) \
		--region $(GCP_REGION) \
		--platform managed \
		--allow-unauthenticated \
		--memory 4Gi \
		--cpu 2 \
		--project $(GCP_PROJECT_ID)
