# Deploying the Inference Endpoint (Docker & Hugging Face)

This guide explains how to build a Docker image for the FastAPI endpoint and how to deploy it to Hugging Face Inference Endpoints or run locally.

Build locally (with GPU support):

1. Build the image:

   docker build -t amchi_asr_inference:latest deploy/

2. Run (exposes port 8000):

   docker run --gpus all -p 8000:8000 -e MODEL_ID=milind-kopikar/amchi_asr-mms -it amchi_asr_inference:latest

Hugging Face Inference Endpoints
-------------------------------
You can provide a custom container to HF Inference Endpoints. Steps:

1. Tag and push container to a registry (ECR/GCR/DockerHub).
2. On Hugging Face, create a new Inference Endpoint and select "Custom container" and provide the image URL.
3. Configure instance size (GPU recommended for MMS). Use a machine with enough GPU memory.

Alternative: Push the model checkpoint to the Hugging Face Hub and use a pre-built HF endpoint, or use a Space (Gradio) for demos.

Security & production notes
- Use auth tokens (HF API, service accounts) for private models.
- Add TLS/HTTPS and rate limiting when exposing a public endpoint.
- Add logging, monitoring, and request timeouts.

