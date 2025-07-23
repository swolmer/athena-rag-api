#!/usr/bin/env python3
"""
🚀 ASPS Medical AI Chatbot - Simple Deployment Script
=====================================================

This script simply runs the ASPS Medical AI system.
All the heavy lifting (GitHub downloads, FAISS building, etc.) 
is handled by demo_asps.py main() function.
"""

import subprocess
import os
import sys

def main():
    """Simple deployment - just run the main system"""
    print("🚀 ASPS MEDICAL AI CHATBOT - SIMPLE DEPLOYMENT")
    print("=" * 60)
    print("🎯 Starting ASPS Medical AI System...")
    print("   📥 Git Clone Integration:        Using local nav1.json + nav2.json + navigation_training_data.json")
    print("   📚 Clinical Index:               nav1.json (31,893 chunks) + clinical training directories")
    print("   🧭 Navigation Index:             nav2.json (14,649 chunks) + navigation training data")
    print("   🤖 AI Model:                     Mistral-7B with dual FAISS indexes")
    print("   🌐 Web Interface:                Available on RunPod HTTP Service port")
    print("")
    
    # Set environment variables for RunPod
    os.environ["HF_HOME"] = "/workspace/huggingface_cache"
    print(f"🔧 Environment Setup:")
    print(f"   HF_HOME =                        {os.environ['HF_HOME']}")
    
    # Quick check for local JSON files
    json_files = ["nav1.json", "nav2.json", ""]
    found_files = [f for f in json_files if os.path.exists(f)]
    
    print(f"📊 Knowledge Base Status:")
    if found_files:
        print(f"   ✅ Found {len(found_files)} local JSON files:")
        for file in found_files:
            file_size = os.path.getsize(file) / (1024 * 1024) if os.path.exists(file) else 0
            print(f"      - {file:<35} ({file_size:.1f} MB)")
    else:
        print("   ⚠️  No local JSON files found    - system will use fallback content")
    
    print(f"🚀 RunPod Deployment Info:")
    print(f"   📡 HTTP Service Port:            19524 (configured in RunPod)")
    print(f"   🔗 Access URL:                   Use RunPod's HTTP Service link")
    print(f"   💻 Web Interface:                FastAPI + Interactive Chatbot")
    
    # Run the main system - everything else is handled in demo_asps.py
    print("\n" + "=" * 60)
    print("🚀 Starting demo_asps_1.py (handles all setup automatically)...")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, "demo_asps.py"], 
                              check=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ ASPS Medical AI System completed successfully!")
            print("🌐 Access your chatbot via the RunPod HTTP Service URL")
        else:
            print(f"\n⚠️ System exited with code {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n👋 System interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running system: {e}")

if __name__ == "__main__":
    main()