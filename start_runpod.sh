#!/bin/bash
# Athen.ai Healthcare RAG Platform - RunPod Deployment Script

echo "🏥 Starting Athen.ai Healthcare RAG Platform on RunPod..."
echo "=================================================="

# Set environment variables
export RAG_API_KEY=${RAG_API_KEY:-"kilment1234"}
export HF_TOKEN=${HF_TOKEN:-"your_huggingface_token"}
export ATHEN_JWT_TOKEN=${ATHEN_JWT_TOKEN:-"kilment1234"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:512"}

echo "✅ Environment variables configured"
echo "   RAG_API_KEY: ${RAG_API_KEY:0:8}..."
echo "   HF_TOKEN: ${HF_TOKEN:0:8}..."
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Create necessary directories
mkdir -p /app/data/uploads
mkdir -p /app/data/models
mkdir -p /app/data/indexes
mkdir -p /app/logs

echo "✅ Directories created"

# Install system dependencies if needed
if ! command -v curl &> /dev/null; then
    echo "📦 Installing system dependencies..."
    apt-get update && apt-get install -y curl
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  nvidia-smi not found. Running in CPU mode."
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "✅ Dependencies installed"

# Download required NLTK data
echo "📚 Downloading NLTK data..."
python -c "
import nltk
import os
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
print('✅ NLTK data downloaded')
"

# Set up Hugging Face cache
if [ "$HF_TOKEN" != "your_huggingface_token" ]; then
    echo "🤗 Configuring Hugging Face..."
    python -c "
from huggingface_hub import login
import os
token = os.getenv('HF_TOKEN')
if token and token != 'your_huggingface_token':
    login(token)
    print('✅ Hugging Face authenticated')
else:
    print('⚠️  HF_TOKEN not configured')
"
fi

# Health check function
health_check() {
    echo "🔍 Performing health check..."
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ Health check passed"
            return 0
        fi
        echo "⏳ Attempt $attempt/$max_attempts - waiting for server..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ Health check failed after $max_attempts attempts"
    return 1
}

# Start the application
echo "🚀 Starting Athen.ai Healthcare RAG Platform..."
echo "   FastAPI: http://localhost:8000"
echo "   Streamlit: http://localhost:8501"
echo "   API Docs: http://localhost:8000/docs"

# Run the main application
python main_script_orgid.py &
MAIN_PID=$!

# Wait a moment and check if the process is still running
sleep 5
if ! kill -0 $MAIN_PID 2>/dev/null; then
    echo "❌ Main application failed to start"
    exit 1
fi

# Perform health check
if health_check; then
    echo "🎉 Athen.ai Healthcare RAG Platform is running successfully!"
    echo "=================================================="
    echo "📊 Dashboard: http://localhost:8501"
    echo "📖 API Documentation: http://localhost:8000/docs"
    echo "🔧 Health Check: http://localhost:8000/health"
    echo "=================================================="
else
    echo "❌ Platform failed to start properly"
    kill $MAIN_PID 2>/dev/null
    exit 1
fi

# Keep the script running
wait $MAIN_PID
