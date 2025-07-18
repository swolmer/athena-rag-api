import os
import subprocess
import time
import requests
import sys

# ------------------------------
# 1. Environment Variables
# ------------------------------
print("🛠️ Setting environment variables...")
os.environ["RAG_API_KEY"] = os.getenv("RAG_API_KEY", "kilment1234")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "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")
os.environ["ATHEN_JWT_TOKEN"] = os.getenv("ATHEN_JWT_TOKEN", "kilment1234")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
print(f"✅ RAG_API_KEY: {os.environ['RAG_API_KEY'][:6]}... • CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

# ------------------------------
# 2. Install Requirements
# ------------------------------
print("📦 Installing Python dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--force-reinstall"], check=True)

# ------------------------------
# 3. NLTK Setup
# ------------------------------
print("📚 Downloading NLTK data...")
import nltk
nltk.download("punkt", quiet=True)
print("✅ NLTK ready")

# ------------------------------
# 4. Hugging Face Login
# ------------------------------
print("🤗 Logging into Hugging Face...")
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
print("✅ Hugging Face authenticated")

# ------------------------------
# 5. Run Main Script
# ------------------------------
print("🚀 Launching Athen.ai Platform...")
print("   FastAPI on http://localhost:8000")
print("   Streamlit on http://localhost:8501")

main_process = subprocess.Popen([sys.executable, "main_script_orgid.py"])

# ------------------------------
# 6. Health Check
# ------------------------------
print("🔍 Checking /health endpoint...")
for attempt in range(30):
    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        if r.status_code == 200:
            print("✅ Health check passed")
            break
    except Exception:
        print(f"⏳ Waiting for app (attempt {attempt+1})...")
        time.sleep(2)
else:
    print("❌ App failed health check")
    main_process.kill()
    sys.exit(1)

# ------------------------------
# 7. Ready!
# ------------------------------
print("🎉 Athen.ai is live!")
print("📖 API Docs: http://localhost:8000/docs")
print("📊 Streamlit: http://localhost:8501")
print("🔧 Health: http://localhost:8000/health")
print("==============================================")

# ------------------------------
# 8. Keep Alive
# ------------------------------
try:
    main_process.wait()
except KeyboardInterrupt:
    print("🛑 Shutting down...")
    main_process.terminate()
