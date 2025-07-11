import subprocess
import sys

def run_command(cmd):
    print(f"\n➡️ Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, text=True)
    print(result.stdout if result.stdout else "")

def main():
    # 1. Navigate to /workspace and remove old repo
    run_command("cd /workspace && rm -rf athena-rag-api")

    # 2. Clone the repo on the asps_demo branch with your token
    clone_url = "https://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@github.com/swolmer/athena-rag-api.git"
    run_command(f"git clone -b asps_demo {clone_url} /workspace/athena-rag-api")

    # 3. Change directory to the repo folder
    run_command("cd /workspace/athena-rag-api")

    # 4. List run_all.py to verify it exists
    run_command("ls -l /workspace/athena-rag-api/run_all.py")

    # 5. Run the main script
    run_command(f"{sys.executable} /workspace/athena-rag-api/run_all.py")

    print("\n✅ All done!")

if __name__ == "__main__":
    main()
