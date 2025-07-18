# Athen.ai Healthcare RAG Platform - RunPod Optimized
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV RAG_API_KEY=kilment1234
ENV HF_TOKEN=your_huggingface_token
ENV ATHEN_JWT_TOKEN=kilment1234

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/models /app/data/indexes /app/logs

# Set permissions
RUN chmod +x /app/main_script_orgid.py

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "main_script_orgid.py"]
