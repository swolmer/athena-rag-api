import subprocess
import os
import sys

def run_command(cmd):
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, text=True)
    print(result)

def main():
    # 1. Install dependencies
    run_command("pip install -r requirements.txt")

    # 2. Set Hugging Face cache dir (optional optimization)
    os.environ["HF_HOME"] = "/workspace/huggingface_cache"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # 3. Run the real script (main_script_orgid.py)
    script_path = os.path.join(os.getcwd(), "main_script_orgid.py")
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)

    run_command(f"{sys.executable} {script_path}")

if __name__ == "__main__":
    main()
