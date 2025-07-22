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
    print("   📥 Git Clone Integration: Using local nav1.json + nav2.json + navigation_training_data.json")
    print("   📚 Clinical Index: nav1.json (31,893 chunks) + clinical training directories")
    print("   🧭 Navigation Index: nav2.json (14,649 chunks) + navigation training data")
    print("   🤖 Powered by Mistral-7B with dual FAISS indexes")
    print("")
    
    # Set environment variables for RunPod
    os.environ["HF_HOME"] = "/workspace/huggingface_cache"
    print(f"🔧 Set HF_HOME to: {os.environ['HF_HOME']}")
    
    # Quick check for local JSON files
    json_files = ["nav1.json", "nav2.json", "navigation_training_data.json"]
    found_files = [f for f in json_files if os.path.exists(f)]
    
    if found_files:
        print(f"✅ Found {len(found_files)} local JSON files: {', '.join(found_files)}")
    else:
        print("⚠️ No local JSON files found - system will use fallback content")
    
    
    # Run the main system - everything else is handled in demo_asps.py
    print("\n🚀 Starting demo_asps.py (handles all setup automatically)...")
    
    try:
        result = subprocess.run([sys.executable, "demo_asps.py"], 
                              check=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ ASPS Medical AI System completed successfully!")
        else:
            print(f"\n⚠️ System exited with code {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n👋 System interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running system: {e}")

if __name__ == "__main__":
    main()