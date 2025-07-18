# 🚀 RunPod Deployment Guide - Athen.ai Healthcare RAG Platform

## 📋 Quick Start Checklist

### ✅ Pre-Deployment
- [ ] Fork this repository to your GitHub account
- [ ] Update environment variables in `.env` file
- [ ] Ensure you have a Hugging Face token
- [ ] Have your API keys ready (`kilment1234` is set as default)

### ✅ RunPod Deployment Options

## 🔥 Option 1: One-Click Docker Deploy

1. **Create RunPod Account**: Sign up at [runpod.io](https://runpod.io)

2. **Deploy with Docker**:
   ```bash
   docker run -p 8000:8000 -p 8501:8501 \
     -e RAG_API_KEY=kilment1234 \
     -e HF_TOKEN=your_huggingface_token \
     -e ATHEN_JWT_TOKEN=kilment1234 \
     --gpus all \
     athenai/healthcare-rag:latest
   ```

3. **Access the Platform**:
   - API: `https://your-runpod-url.com:8000`
   - Dashboard: `https://your-runpod-url.com:8501`
   - Docs: `https://your-runpod-url.com:8000/docs`

## ⚙️ Option 2: Manual GitHub Deploy

1. **Clone Repository**:
   ```bash
   git clone https://github.com/yourusername/athen-ai
   cd athen-ai
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

4. **Run Application**:
   ```bash
   chmod +x start_runpod.sh
   ./start_runpod.sh
   ```

## 📊 Option 3: Docker Compose

1. **Start with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

2. **Check Status**:
   ```bash
   docker-compose ps
   docker-compose logs athen-ai
   ```

## 🌐 Integration with Your Website

### Frontend JavaScript Integration

```javascript
// Configure your RunPod endpoint
const ATHEN_API_BASE = "https://your-runpod-url.com";
const API_TOKEN = "kilment1234";

// Upload documents
const uploadDocs = async (files, orgId) => {
  const formData = new FormData();
  formData.append('file', files[0]);
  
  const response = await fetch(`${ATHEN_API_BASE}/rag/${orgId}/upload`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_TOKEN}`
    },
    body: formData
  });
  return response.json();
};

// Start training
const startTraining = async (orgId) => {
  const response = await fetch(`${ATHEN_API_BASE}/rag/${orgId}/training/start`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_TOKEN}`,
      'Content-Type': 'application/json'
    }
  });
  return response.json();
};

// Query chatbot
const queryChatbot = async (orgId, question) => {
  const response = await fetch(`${ATHEN_API_BASE}/rag/${orgId}/chat`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_TOKEN}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ question })
  });
  return response.json();
};
```

### React Component Example

```jsx
import React, { useState } from 'react';

const AthenaraiChat = ({ orgId }) => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleQuery = async () => {
    setLoading(true);
    try {
      const result = await queryChatbot(orgId, question);
      setResponse(result.answer);
    } catch (error) {
      console.error('Chat error:', error);
    }
    setLoading(false);
  };

  return (
    <div className="athen-ai-chat">
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a medical question..."
      />
      <button onClick={handleQuery} disabled={loading}>
        {loading ? 'Thinking...' : 'Ask AI'}
      </button>
      {response && (
        <div className="response">
          {response}
        </div>
      )}
    </div>
  );
};

export default AthenaraiChat;
```

## 🔧 Environment Configuration

### Required Environment Variables

```bash
# API Authentication
RAG_API_KEY=kilment1234
ATHEN_JWT_TOKEN=kilment1234
HF_TOKEN=your_huggingface_token

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Application Settings
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO
```

### Optional Configuration

```bash
# Database (optional)
DATABASE_URL=sqlite:///./athen_ai.db

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Model Configuration
DEFAULT_MODEL=microsoft/DialoGPT-medium
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MAX_TOKENS=512
TEMPERATURE=0.7
```

## 📡 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/rag/{org_id}/upload` | POST | Upload documents |
| `/rag/{org_id}/training/start` | POST | Start training |
| `/rag/{org_id}/training/status` | GET | Training status |
| `/rag/{org_id}/chat` | POST | Query chatbot |
| `/dashboard/metrics` | GET | Platform metrics |

### Authentication

All endpoints require Bearer token authentication:

```bash
curl -H "Authorization: Bearer kilment1234" \
     https://your-runpod-url.com/health
```

## 🚀 Performance Optimization

### GPU Requirements

- **Minimum**: NVIDIA GTX 1080 (8GB VRAM)
- **Recommended**: NVIDIA RTX 3080/4080 (12GB+ VRAM)
- **Optimal**: NVIDIA A100/H100 (40GB+ VRAM)

### Memory Configuration

```bash
# For 8GB GPU
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For 16GB+ GPU
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# For 32GB+ GPU
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048
```

## 🔍 Monitoring & Debugging

### Health Checks

```bash
# API health
curl https://your-runpod-url.com/health

# Detailed metrics
curl -H "Authorization: Bearer kilment1234" \
     https://your-runpod-url.com/dashboard/metrics
```

### Logs

```bash
# Application logs
tail -f logs/athen_ai.log

# Docker logs
docker logs -f athen-ai

# RunPod logs
runpod logs
```

### Common Issues

1. **GPU Not Detected**:
   ```bash
   nvidia-smi  # Check GPU availability
   export CUDA_VISIBLE_DEVICES=0
   ```

2. **Memory Issues**:
   ```bash
   # Reduce batch size or model size
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
   ```

3. **Port Conflicts**:
   ```bash
   # Change ports in docker-compose.yml
   ports:
     - "8001:8000"  # FastAPI
     - "8502:8501"  # Streamlit
   ```

## 📞 Support

- 📖 [Full Documentation](./docs/)
- 🐛 [Issue Tracker](https://github.com/yourusername/athen-ai/issues)
- 💬 [Discord Community](https://discord.gg/athen-ai)
- 📧 [Email Support](mailto:support@athen.ai)

## 🎯 Production Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database backups enabled
- [ ] Monitoring alerts set up
- [ ] HIPAA compliance verified
- [ ] API rate limiting configured
- [ ] Log rotation enabled
- [ ] Security audit completed

---

**🚀 Ready for Production! Your healthcare AI platform is now live on RunPod.**
