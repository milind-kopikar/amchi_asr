# Minimal Dockerfile for the FastAPI HF inference endpoint
# Build: docker build -t amchi_asr_inference:latest .
# Run: docker run --gpus all -p 8000:8000 -e MODEL_ID=milind-kopikar/amchi_asr-mms -it amchi_asr_inference:latest

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn8-runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y ffmpeg git curl && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python deps (pinning recommended in real deployments)
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir transformers==4.57.3 accelerate jiwer soundfile fastapi uvicorn python-multipart aiofiles

# Expose port and start uvicorn
ENV MODEL_ID=facebook/mms-1b-all
EXPOSE 8000
CMD ["uvicorn", "scripts.hf_inference_endpoint:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
