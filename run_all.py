#!/usr/bin/env python3
"""
🚀 ASPS Medical AI Chatbot - RunPod Deployment Script
====================================================

Optimized for RunPod with pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22
Handles complete setup and deployment of the ASPS Medical AI system.
"""

import subprocess
import os
import sys
import time
from pathlib import Path

def print_runpod_info():
    """Print RunPod-specific deployment information"""
    print("🚀 ASPS MEDICAL AI CHATBOT - RUNPOD DEPLOYMENT")
    print("=" * 65)
    print("�️ RunPod Configuration Detected:")
    print(f"   🐋 Container:                    pytorch:2.1.0-py3.10-cuda11.8.0")
    print(f"   � Workspace Mount:              /workspace")
    print(f"   🌐 HTTP Port:                    19524 (configured)")
    print(f"   🔑 SSH Port:                     22 (configured)")
    print(f"   🐍 Python Version:               {sys.version.split()[0]}")
    print(f"   📂 Current Directory:            {os.getcwd()}")
    print("")

def setup_runpod_environment():
    """Setup RunPod-specific environment variables and directories"""
    print("🔧 Setting up RunPod environment...")
    
    # RunPod-specific environment variables
    env_vars = {
        "HF_HOME": "/workspace/huggingface_cache",
        "TRANSFORMERS_CACHE": "/workspace/transformers_cache",
        "TORCH_HOME": "/workspace/torch_cache",
        "PYTHONPATH": f"{os.getcwd()}:/workspace",
        "PORT": "19524"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   ✅ {key:<20} = {value}")
    
    # Create necessary directories
    dirs_to_create = [
        "/workspace/huggingface_cache",
        "/workspace/transformers_cache", 
        "/workspace/torch_cache",
        "org_data/asps",
        "logs"
    ]
    
    print("\n📁 Creating directories...")
    for directory in dirs_to_create:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
        except Exception as e:
            print(f"   ⚠️ {directory} - {e}")

def check_gpu_status():
    """Check GPU availability and CUDA setup"""
    print("\n🧠 Checking GPU status...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"   ✅ CUDA Available:               {torch.cuda.is_available()}")
            print(f"   🔢 GPU Count:                    {gpu_count}")
            print(f"   📛 GPU Name:                     {gpu_name}")
            print(f"   💾 GPU Memory:                   {gpu_memory:.1f} GB")
            
            # Set optimal settings based on GPU
            if any(gpu in gpu_name.lower() for gpu in ["4090", "a100", "v100", "a6000"]):
                print("   🚀 High-end GPU detected - optimal performance expected")
            else:
                print("   ⚡ GPU detected - good performance expected")
        else:
            print("   ⚠️ No CUDA GPU detected - using CPU mode")
            print("   💡 Performance will be limited without GPU")
    except ImportError:
        print("   ❌ PyTorch not found - cannot check GPU status")

def install_requirements():
    """Install required packages if not present"""
    print("\n📦 Checking Python requirements...")
    
    try:
        # Try importing key packages
        import torch
        import transformers
        import fastapi
        import uvicorn
        print("   ✅ Core packages already installed")
        return True
    except ImportError:
        print("   ⚠️ Missing packages detected - installing requirements...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, capture_output=True, text=True)
            print("   ✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install requirements: {e}")
            print(f"   📋 Error output: {e.stderr}")
            return False

def check_knowledge_bases():
    """Check for knowledge base files"""
    print("\n📚 Checking knowledge base files...")
    
    kb_files = [
        "nav1.json",
        "nav2.json", 
        "navigation_training_data.json"
    ]
    
    found_files = 0
    for file in kb_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"   ✅ {file:<35} ({size:.1f} MB)")
            found_files += 1
        else:
            print(f"   📥 {file:<35} Will be downloaded from GitHub")
    
    if found_files >= 2:
        print(f"   🎯 Found {found_files}/3 knowledge bases - system ready!")
    else:
        print(f"   📡 Will download missing files during startup")
    
    return True

def main():
    """Main deployment function for RunPod"""
    print_runpod_info()
    
    # Step 1: Setup RunPod environment
    setup_runpod_environment()
    
    # Step 2: Check GPU status
    check_gpu_status()
    
    # Step 3: Install requirements if needed
    if not install_requirements():
        print("\n❌ Failed to install requirements - deployment aborted")
        return 1
    
    # Step 4: Check knowledge bases
    check_knowledge_bases()
    
    # Step 5: Start the main system
    print("\n" + "=" * 65)
    print("🚀 Starting ASPS Medical AI System...")
    print("   🎯 Dual Knowledge System:        Clinical + Navigation")
    print("   🤖 AI Model:                     Mistral-7B") 
    print("   🌐 Web Interface:                FastAPI + Interactive Chat")
    print("   📡 Access URL:                   Use RunPod's HTTP Service link")
    print("=" * 65)
    
    try:
        print("🔄 Importing FastAPI app from demo_asps.py...")
        import demo_asps
        from uvicorn import Config, Server
        port = int(os.environ.get("PORT", 19524))
        config = Config(app=demo_asps.app, host="0.0.0.0", port=port, log_level="info")
        server = Server(config)
        print(f"🌐 Starting ASPS Medical AI Chatbot on http://0.0.0.0:{port}")
        print(f"📊 Health endpoint: http://0.0.0.0:{port}/health")
        print(f"📚 API docs: http://0.0.0.0:{port}/docs")
        server.run()
        print("✅ ASPS Medical AI System completed successfully!")
    except KeyboardInterrupt:
        print("\n👋 Deployment interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error during deployment: {e}")
        return 1
    return 0

if __name__ == "__main__":
    main()
