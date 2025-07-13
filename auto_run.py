import subprocess
import sys
import os

def run_command(cmd):
    print(f"\n➡️ Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
    print(result.stdout if result.stdout else "")

def main():
    # ✅ Define repo path and script
    repo_path = "/workspace/athena-rag-api"
    script_path = os.path.join(repo_path, "main_script_orgid.py")

    # ✅ 1. Remove old repo if it exists
    if os.path.exists(repo_path):
        run_command(f"rm -rf {repo_path}")

    # ✅ 2. Clone your GitHub repo with your provided token
    clone_url = (
        "https://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"
        "@github.com/swolmer/athena-rag-api.git"
    )
    run_command(f"git clone -b asps_demo {clone_url} {repo_path}")

    # ✅ 3. Confirm script exists
    run_command(f"ls -l {script_path}")

    # ✅ 4. Run main_script_orgid.py
    run_command(f"{sys.executable} {script_path}")

    print("\n✅ Successfully ran main_script_orgid.py!")

if __name__ == "__main__":
    main()
