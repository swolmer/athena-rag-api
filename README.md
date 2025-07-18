# 🏥 Athen.ai - Healthcare RAG Chatbot Platform

[![Deploy on RunPod](https://img.shields.io/badge/Deploy-RunPod-blue)](https://runpod.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-green)](https://docker.com)
[![API](https://img.shields.io/badge/API-Complete-orange)](./API_helper_complete.py)

## 🎯 **One-Click Deployment to RunPod**

### **Quick Start (3 Steps)**
1. **Fork this repository** on GitHub
2. **Deploy to RunPod** using our Docker template 
3. **Connect to your website** using the API endpoints

---

## 📋 **System Overview**

**Athen.ai** is a production-ready healthcare AI platform that allows:
- 📁 **Document Upload**: Clients upload medical documentation 
- 🤖 **Async Training**: RAG models train automatically on RunPod
- 💬 **Instant Chatbot**: Trained chatbots deploy to your website
- 🏥 **Multi-Organization**: Separate models for different hospitals
- 📊 **Analytics Dashboard**: Real-time performance monitoring

---

## 🐳 **RunPod Deployment**

### **Method 1: One-Click Docker Deploy**
```bash
# RunPod will automatically run this
docker run -p 8000:8000 -p 8501:8501 \
  -e RAG_API_KEY=kilment1234 \
  -e HF_TOKEN=your_hf_token \
  athenai/healthcare-rag:latest
```

### **Method 2: Manual Setup**
```bash
git clone https://github.com/yourusername/athen-ai
cd athen-ai
pip install -r requirements.txt
python main_script_orgid.py
```

---

## 🌐 **Website Integration**

### **Frontend JavaScript (React/Next.js)**
```javascript
// Connect to your RunPod deployment
const ATHEN_API_BASE = "https://your-runpod-url.com";

// Upload documents
const uploadDocs = async (files, orgId) => {
  const formData = new FormData();
  formData.append('file', files[0]);
  formData.append('org_id', orgId);
  
  const response = await fetch(`${ATHEN_API_BASE}/rag/${orgId}/upload`, {
    method: 'POST',
    body: formData
  });
  return response.json();
};

// Start training
const startTraining = async (orgId) => {
  const response = await fetch(`${ATHEN_API_BASE}/rag/${orgId}/training/start`, {
    method: 'POST'
  });
  return response.json();
};

// Query chatbot
const queryChatbot = async (orgId, question) => {
  const response = await fetch(`${ATHEN_API_BASE}/rag/${orgId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  return response.json();
};
```

### **Complete Dashboard Integration**
- 📊 **Performance Dashboard**: Real-time metrics
- 📁 **Active Projects**: Project management 
- 🏥 **Templates**: Medical specialty templates
- 📊 **Analytics**: Usage statistics
- 👥 **Collaboration**: Team management
- 📂 **Versioning**: Compliance & audit trails

---

## 🔧 **Environment Setup**

### **Required Environment Variables**
```bash
# Create .env file
RAG_API_KEY=kilment1234
HF_TOKEN=your_huggingface_token
ATHEN_API_BASE_URL=https://api.athen.ai/v1
ATHEN_JWT_TOKEN=kilment1234
```

### **RunPod Environment**
```bash
# GPU Requirements
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## 📚 **API Documentation**

### **Core Endpoints**
- `POST /rag/{org_id}/upload` - Upload training documents
- `POST /rag/{org_id}/training/start` - Start model training
- `GET /rag/{org_id}/training/status` - Check training status
- `POST /rag/{org_id}/chat` - Query trained chatbot
- `GET /dashboard/metrics` - Get performance metrics

### **Dashboard Endpoints**
- `GET /projects` - List all projects
- `GET /templates` - Get medical templates
- `GET /analytics/specialty/{specialty}/models` - Specialty models
- `POST /projects/{project_id}/collaborators` - Add team members

[📖 **Complete API Documentation**](./docs/API.md)

---

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Client Web    │───▶│   RunPod     │───▶│   Healthcare    │
│   Dashboard     │    │   Server     │    │   RAG Models    │
└─────────────────┘    └──────────────┘    └─────────────────┘
        │                       │                     │
        │              ┌─────────▼──────────┐        │
        │              │   API Endpoints    │        │
        │              │ • Upload           │        │
        │              │ • Training         │        │
        │              │ • Chat             │        │
        │              │ • Analytics        │        │
        │              └────────────────────┘        │
        │                                            │
        └──────────────── Async Training ────────────┘
```

---

## 🚀 **Production Features**

### ✅ **Complete Healthcare Platform**
- 🏥 Multi-organization support (hospitals, clinics)
- 📄 Document processing (PDF, DOCX, images)
- 🤖 Mistral-7B medical chatbots
- 📊 FAISS vector search
- 🔒 JWT authentication
- 📈 Real-time analytics

### ✅ **RunPod Optimized**
- 🐳 Docker containerized
- ⚡ GPU acceleration
- 🔧 Auto-scaling
- 📡 API-first design
- 🌐 Public endpoints

### ✅ **Website Ready**
- 🎨 React/Next.js compatible
- 📱 Responsive design
- 🔌 REST API integration
- 📊 Dashboard components
- 🚀 Production deployment

---

## 📞 **Support & Documentation**

- 📖 [API Documentation](./docs/API.md)
- 🐳 [Docker Setup](./docs/DOCKER.md)
- 🌐 [Website Integration](./docs/INTEGRATION.md)
- 🚀 [RunPod Deployment](./docs/RUNPOD.md)
- 🏥 [Healthcare Templates](./docs/TEMPLATES.md)

---

## 📄 **License**

MIT License - Built for Healthcare Innovation

---

**🎯 Ready for Production! Deploy to RunPod in minutes and connect to your website instantly.**
