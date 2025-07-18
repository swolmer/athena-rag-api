#!/bin/bash
# Athen.ai Healthcare RAG Platform - RunPod Deployment Script

set -e

echo "🏥 Starting Athen.ai Healthcare RAG Platform on RunPod..."
echo "=================================================="

# ---------------------------
# 1. Environment variables
# ---------------------------
export RAG_API_KEY=${RAG_API_KEY:-"kilment1234"}
export HF_TOKEN=${HF_TOKEN:-"your_huggingface_token"}
export ATHEN_JWT_TOKEN=${ATHEN_JWT_TOKEN:-"kilment1234"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:512"}

echo "✅ Environment variables configured"
echo "   RAG_API_KEY: ${RAG_API_KEY:0:8}..."
echo "   HF_TOKEN: ${HF_TOKEN:0:8}..."
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ---------------------------
# 2. Create necessary folders
# ---------------------------
mkdir -p /app/data/uploads /app/data/models /app/data/indexes /app/logs
echo "✅ Directories created"

# ---------------------------
# 3. Check GPU
# ---------------------------
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  nvidia-smi not found. Running in CPU mode."
fi

# ---------------------------
# 4. Install Python dependencies
# ---------------------------
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt --no-deps || true

# Manually install important stragglers (if needed)
pip install --no-cache-dir nltk pandas || true

echo "✅ Dependencies installed"

# ---------------------------
# 5. Download NLTK data
# ---------------------------
echo "📚 Downloading NLTK data..."
python - <<EOF
import nltk, os
try:
    nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
    nltk.download('punkt', download_dir=nltk_data_path)
    print("✅ NLTK data downloaded")
except Exception as e:
    print(f"⚠️ NLTK download failed: {e}")
EOF

# ---------------------------
# 6. Hugging Face login (optional)
# ---------------------------
if [ "$HF_TOKEN" != "your_huggingface_token" ]; then
    echo "🤗 Logging in to Hugging Face..."
    python -c "
from huggingface_hub import login
import os
token = os.getenv('HF_TOKEN')
if token:
    login(token)
    print('✅ Hugging Face authenticated')
else:
    print('⚠️ HF_TOKEN not configured')
"
fi

# ---------------------------
# 7. Health Check Function
# ---------------------------
health_check() {
    echo "🔍 Performing health check..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health &>/dev/null; then
            echo "✅ Health check passed"
            return 0
        fi
        echo "⏳ Attempt $i - waiting for server..."
        sleep 2
    done
    echo "❌ Health check failed"
    return 1
}

# ---------------------------
# 8. Start App
# ---------------------------
echo "🚀 Starting Athen.ai Healthcare RAG Platform..."
echo "   FastAPI: http://localhost:8000"
echo "   Streamlit: http://localhost:8501"
echo "   API Docs: http://localhost:8000/docs"

python main_script_orgid.py &
MAIN_PID=$!

sleep 5
if ! kill -0 $MAIN_PID 2>/dev/null; then
    echo "❌ Main application failed to start"
    exit 1
fi

# ---------------------------
# 9. Verify App is Running
# ---------------------------
if health_check; then
    echo "🎉 Athen.ai Healthcare RAG Platform is running successfully!"
    echo "=================================================="
    echo "📊 Dashboard: http://localhost:8501"
    echo "📖 API Documentation: http://localhost:8000/docs"
    echo "🔧 Health Check: http://localhost:8000/health"
    echo "=================================================="
else
    echo "❌ Platform failed health check"
    kill $MAIN_PID 2>/dev/null
    exit 1
fi

# ---------------------------
# 10. Keep Script Running
# ---------------------------
wait $MAIN_PID
