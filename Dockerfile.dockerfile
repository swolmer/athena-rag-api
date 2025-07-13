# Base image with CUDA 12.1, Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and essential system tools
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    tesseract-ocr poppler-utils build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    git curl wget unzip ffmpeg \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r common/requirements.txt

# Download NLTK punkt tokenizer used by your script
RUN python -m nltk.downloader --download_dir=/usr/share/nltk_data punkt
ENV NLTK_DATA=/usr/share/nltk_data

# Run the RAG pipeline
CMD ["python", "main_script_orgid.py"]
