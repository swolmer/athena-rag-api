import subprocess
import os
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, text=True)
    print(result)

def main():
    # 1. Install dependencies
    run_command("pip install -r requirements.txt")

    # 2. Set Hugging Face cache env var for this process
    os.environ["HF_HOME"] = "/workspace/huggingface_cache"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # 3. Run your main Python script
    run_command(f"{sys.executable} demo_asps.py")

if __name__ == "__main__":
    main()
