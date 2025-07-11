# Base image with CUDA 12.1, Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python & utilities
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    tesseract-ocr poppler-utils build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    git curl wget unzip ffmpeg && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    pip install --upgrade pip

# Copy code into container
WORKDIR /app
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir -r common/requirements.txt

# Download NLTK punkt
RUN python -m nltk.downloader punkt

# Run the chatbot as CLI or API
CMD ["python", "main.py"]
