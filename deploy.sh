#!/bin/bash
# ===========================================
# 🚀 ATHEN.AI RAG API - RUNPOD DEPLOYMENT SCRIPT
# ===========================================

echo "🚀 Starting Athen.ai RAG API deployment..."

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env file with your actual tokens!"
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True)"

# Create org_data directory
echo "📁 Creating organization data directory..."
mkdir -p org_data

# Check GPU availability
echo "🧠 Checking GPU availability..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

echo "✅ Deployment setup complete!"
echo ""
echo "🌐 To start the API server, run:"
echo "python main_script_orgid.py"
echo ""
echo "📡 Your API will be available at:"
echo "https://your-runpod-id.proxy.runpod.net"
echo ""
echo "📚 API Documentation:"
echo "https://your-runpod-id.proxy.runpod.net/docs"
