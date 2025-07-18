# 🚀 Athen.ai RunPod Deployment Package

## 📦 **What's in this folder:**

### **Essential Files for RunPod Deployment:**

1. **`README.md`** - Complete documentation with kilment1234 API keys configured
2. **`Dockerfile`** - GPU-optimized container for RunPod deployment
3. **`requirements.txt`** - All Python dependencies including GPU support
4. **`main_script_orgid.py`** - Production FastAPI server with healthcare RAG
5. **`dashboard.py`** - Streamlit web dashboard interface
6. **`.env`** - Environment configuration (API keys: kilment1234)
7. **`docker-compose.yml`** - Local development setup
8. **`start_runpod.sh`** - Automated deployment script for RunPod
9. **`runpod-template.json`** - RunPod deployment template
10. **`RUNPOD_DEPLOY.md`** - Complete deployment guide
11. **`.gitignore`** - Git ignore rules

## 🎯 **How to Upload to GitHub:**

### **Option 1: Create New Branch (Recommended)**
1. Go to your GitHub repo: `https://github.com/swolmer/athena-rag-api`
2. Click **"main"** dropdown → **"Create new branch"**
3. Name it: `runpod-deployment`
4. Upload all files from this folder to the new branch

### **Option 2: Replace Main Branch**
1. Go to your GitHub repo
2. Upload these files to replace the existing ones
3. All API keys are already configured with kilment1234

## 🚀 **Deployment Steps:**

1. **Upload files** to GitHub (either method above)
2. **Go to RunPod.io** and create account
3. **Deploy using Docker**: 
   ```bash
   docker run -p 8000:8000 -p 8501:8501 \
     -e RAG_API_KEY=kilment1234 \
     -e HF_TOKEN=your_hf_token \
     --gpus all \
     your-repo/athena-rag
   ```
4. **Access your platform**:
   - API: `https://your-runpod-url:8000`
   - Dashboard: `https://your-runpod-url:8501`
   - Docs: `https://your-runpod-url:8000/docs`

## ✅ **Ready Features:**

- 🏥 Multi-organization healthcare RAG
- 📁 Document upload and processing
- 🤖 Async GPU-accelerated training
- 💬 Real-time AI chat interface
- 📊 Analytics dashboard
- 🔒 JWT authentication (kilment1234)
- 🐳 Docker containerized
- ⚡ GPU optimized
- 📖 Complete API documentation

---

**🎉 Everything is ready for RunPod deployment!**
