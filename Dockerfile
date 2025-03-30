# Use CUDA-enabled base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TRANSFORMERS_CACHE=/app/models \
    HUGGINGFACE_HUB_CACHE=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create model cache directory
RUN mkdir -p /app/models

# Entrypoint command
CMD ["python3", "bot.py"]
