#!/bin/bash
# Unified Launcher for Athen.ai RAG Platform on RunPod

set -e  # Exit if anything fails

echo "🛠️  Preparing Athen.ai Healthcare RAG Platform..."
echo "=============================================="

# -------------------------------
# 1. Environment Setup
# -------------------------------
export RAG_API_KEY=${RAG_API_KEY:-"kilment1234"}
export HF_TOKEN=${HF_TOKEN:-"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"}
export ATHEN_JWT_TOKEN=${ATHEN_JWT_TOKEN:-"kilment1234"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:512"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

echo "✅ Environment variables set"
echo "   RAG_API_KEY: ${RAG_API_KEY:0:6}... • CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# -------------------------------
# 2. Install Python Requirements
# -------------------------------
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt --no-deps || true
pip install nltk pandas python-dotenv || true

# -------------------------------
# 3. Download NLTK Data
# -------------------------------
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); print('✅ NLTK ready')"

# -------------------------------
# 4. Hugging Face Auth
# -------------------------------
echo "🤗 Logging into Hugging Face with provided token..."
python -c "from huggingface_hub import login; login('$HF_TOKEN')"
echo "✅ Hugging Face authenticated"

# -------------------------------
# 5. Start App
# -------------------------------
echo "🚀 Launching Athen.ai Platform..."
echo "   FastAPI on http://localhost:8000"
echo "   Streamlit on http://localhost:8501"

python main_script_orgid.py &
MAIN_PID=$!

sleep 5

if ! kill -0 $MAIN_PID 2>/dev/null; then
  echo "❌ App failed to start"
  exit 1
fi

# -------------------------------
# 6. Health Check
# -------------------------------
echo "🔍 Checking /health endpoint..."
for i in {1..30}; do
  if curl -s http://localhost:8000/health &> /dev/null; then
    echo "✅ Health check passed"
    break
  else
    echo "⏳ Waiting for app (attempt $i)..."
    sleep 2
  fi
done

# -------------------------------
# 7. Ready!
# -------------------------------
echo "🎉 Athen.ai is live!"
echo "📖 API Docs: http://localhost:8000/docs"
echo "📊 Streamlit: http://localhost:8501"
echo "🔧 Health: http://localhost:8000/health"
echo "=============================================="

# Keep the app running
wait $MAIN_PID
