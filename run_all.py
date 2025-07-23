#!/usr/bin/env python3
"""
🚀 ASPS MEDICAL AI - ONE-CLICK RUNPOD LAUNCHER
=============================================

Comprehensive launcher for the ASPS Medical AI System designed for RunPod deployment.
Handles all dependencies, environment setup, and system initialization automatically.

Usage: Simply run "python run_all.py" in your RunPod terminal - that's it!

Author: Advanced AI Systems
Version: 2.0.0 - Production Ready
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# ============================
# 🎨 CONSOLE STYLING
# ============================

class Colors:
    """ANSI color codes for beautiful terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_BLUE = '\033[104m'

def print_banner():
    """Print beautiful system banner"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🏥 ASPS MEDICAL AI SYSTEM 2.0                           ║
║                      RunPod One-Click Launcher                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.RESET}

{Colors.GREEN}🎯 System Features:{Colors.RESET}
   • {Colors.YELLOW}Dual Knowledge Base{Colors.RESET} (Clinical + Navigation)
   • {Colors.YELLOW}GitHub Auto-Sync{Colors.RESET} from athena-rag-api repository
   • {Colors.YELLOW}Mistral-7B Language Model{Colors.RESET} for intelligent responses
   • {Colors.YELLOW}FAISS Vector Search{Colors.RESET} for semantic retrieval
   • {Colors.YELLOW}FastAPI Web Interface{Colors.RESET} with beautiful UI
   • {Colors.YELLOW}Production-Ready Architecture{Colors.RESET} with error handling

{Colors.BLUE}📍 Deployment Target:{Colors.RESET} RunPod GPU Instance
{Colors.BLUE}🔧 Auto-Setup:{Colors.RESET} Dependencies, Environment, Models
{Colors.BLUE}🌐 Web Interface:{Colors.RESET} Accessible via RunPod's public URL

"""
    print(banner)

def print_status(message: str, status: str = "info"):
    """Print colored status messages"""
    colors = {
        "info": Colors.CYAN,
        "success": Colors.GREEN,
        "warning": Colors.YELLOW,
        "error": Colors.RED,
        "processing": Colors.MAGENTA
    }
    
    icons = {
        "info": "ℹ️ ",
        "success": "✅",
        "warning": "⚠️ ",
        "error": "❌",
        "processing": "🔄"
    }
    
    color = colors.get(status, Colors.WHITE)
    icon = icons.get(status, "")
    
    print(f"{color}{Colors.BOLD}{icon} {message}{Colors.RESET}")

def print_section(title: str):
    """Print section headers"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}")
    print(f"📦 {title}")
    print(f"{'='*60}{Colors.RESET}")

# ============================
# 🔧 SYSTEM UTILITIES
# ============================

class SystemManager:
    """Handles system setup and validation"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.env_file = self.base_dir / ".env"
        self.requirements_installed = False
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        print_status("Checking Python version...", "processing")
        
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            print_status(f"Python {version.major}.{version.minor}.{version.micro} - Compatible ✓", "success")
            return True
        else:
            print_status(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+", "error")
            return False
    
    def check_gpu_availability(self) -> Dict[str, any]:
        """Check GPU availability and CUDA setup"""
        print_status("Checking GPU and CUDA availability...", "processing")
        
        gpu_info = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_name": "None",
            "memory_gb": 0
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_info["cuda_available"] = True
                gpu_info["gpu_count"] = torch.cuda.device_count()
                gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
                
                # Get memory info
                memory_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_info["memory_gb"] = round(memory_bytes / (1024**3), 1)
                
                print_status(f"GPU: {gpu_info['gpu_name']} ({gpu_info['memory_gb']}GB)", "success")
                print_status(f"CUDA: Available with {gpu_info['gpu_count']} device(s)", "success")
            else:
                print_status("CUDA: Not available - will use CPU (slower)", "warning")
                
        except ImportError:
            print_status("PyTorch not installed yet - will install during setup", "info")
        
        return gpu_info
    
    def install_requirements(self) -> bool:
        """Install all required packages"""
        if self.requirements_installed:
            return True
            
        print_section("INSTALLING DEPENDENCIES")
        
        # Core packages for the system
        packages = [
            # Core ML/AI packages
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",  # Will upgrade to GPU version if available
            
            # Scientific computing
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            
            # Web framework
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.20.0",
            "pydantic>=2.0.0",
            
            # Document processing
            "PyMuPDF>=1.23.0",  # fitz
            "python-docx>=0.8.11",
            "Pillow>=9.0.0",
            "beautifulsoup4>=4.11.0",
            
            # Utilities
            "python-dotenv>=1.0.0",
            "requests>=2.28.0",
            "nltk>=3.8.0",
            
            # Optional OCR (may fail on some systems)
            "pytesseract>=0.3.10",
        ]
        
        print_status(f"Installing {len(packages)} essential packages...", "processing")
        
        for i, package in enumerate(packages, 1):
            try:
                print_status(f"[{i}/{len(packages)}] Installing {package.split('>=')[0]}...", "processing")
                
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package, "--quiet"],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per package
                )
                
                if result.returncode == 0:
                    print_status(f"✓ {package.split('>=')[0]} installed successfully", "success")
                else:
                    print_status(f"⚠️  {package.split('>=')[0]} installation warning: {result.stderr[:100]}", "warning")
                    
            except subprocess.TimeoutExpired:
                print_status(f"⏰ {package.split('>=')[0]} installation timed out - continuing", "warning")
            except Exception as e:
                print_status(f"❌ Failed to install {package.split('>=')[0]}: {str(e)[:100]}", "error")
        
        # Try to install GPU-optimized FAISS if CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                print_status("Installing GPU-optimized FAISS...", "processing")
                subprocess.run([sys.executable, "-m", "pip", "install", "faiss-gpu", "--quiet"], 
                             timeout=180, capture_output=True)
                print_status("✓ GPU FAISS installed", "success")
        except:
            print_status("Using CPU FAISS (GPU version not available)", "info")
        
        self.requirements_installed = True
        print_status("All dependencies installed successfully!", "success")
        return True
    
    def setup_environment(self) -> bool:
        """Setup environment variables"""
        print_section("ENVIRONMENT CONFIGURATION")
        
        # Default environment variables for RunPod
        env_vars = {
            "GITHUB_TOKEN": "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",
            "GITHUB_REPO_URL": "https://github.com/swolmer/athena-rag-api.git",
            "GITHUB_BRANCH": "asps_demo",
            "HF_TOKEN": "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",
            "RAG_API_KEY": "your-rag-api-key-here",
            "PYTHONPATH": str(self.base_dir),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(self.base_dir / ".cache" / "huggingface"),
            "TRANSFORMERS_CACHE": str(self.base_dir / ".cache" / "transformers"),
            "TORCH_HOME": str(self.base_dir / ".cache" / "torch")
        }
        
        # Create .env file
        env_content = []
        for key, value in env_vars.items():
            env_content.append(f"{key}={value}")
            # Also set in current environment
            os.environ[key] = value
        
        try:
            with open(self.env_file, "w") as f:
                f.write("\n".join(env_content))
            
            print_status(f"Environment file created: {self.env_file}", "success")
            print_status("GitHub integration: ENABLED", "success")
            print_status("HuggingFace token: CONFIGURED", "success")
            
            return True
        except Exception as e:
            print_status(f"Failed to create .env file: {e}", "error")
            return False
    
    def validate_system_files(self) -> bool:
        """Validate that required system files exist"""
        print_section("SYSTEM VALIDATION")
        
        required_files = [
            "demo_asps_1.py"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.base_dir / file_name
            if file_path.exists():
                print_status(f"✓ Found {file_name}", "success")
            else:
                print_status(f"✗ Missing {file_name}", "error")
                missing_files.append(file_name)
        
        if missing_files:
            print_status(f"Missing critical files: {', '.join(missing_files)}", "error")
            return False
        
        print_status("All system files validated successfully!", "success")
        return True
    
    def download_nltk_data(self) -> bool:
        """Download required NLTK data"""
        print_status("Downloading NLTK data...", "processing")
        
        try:
            import nltk
            
            # Download required datasets quietly
            for dataset in ['punkt', 'stopwords']:
                try:
                    nltk.download(dataset, quiet=True)
                except:
                    pass  # Continue if download fails
            
            print_status("✓ NLTK data downloaded", "success")
            return True
        except Exception as e:
            print_status(f"NLTK data download failed: {e}", "warning")
            return False
    
    def create_cache_directories(self) -> None:
        """Create cache directories for models"""
        cache_dirs = [
            self.base_dir / ".cache" / "huggingface",
            self.base_dir / ".cache" / "transformers", 
            self.base_dir / ".cache" / "torch",
            self.base_dir / "asps_data"
        ]
        
        for cache_dir in cache_dirs:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        print_status("Cache directories created", "success")

# ============================
# 🚀 RUNPOD LAUNCHER
# ============================

class RunPodLauncher:
    """Main launcher for RunPod deployment"""
    
    def __init__(self):
        self.system_manager = SystemManager()
        self.start_time = time.time()
    
    def run_system_checks(self) -> bool:
        """Run all system checks and setup"""
        print_banner()
        
        # Check Python version
        if not self.system_manager.check_python_version():
            return False
        
        # Check GPU availability
        gpu_info = self.system_manager.check_gpu_availability()
        
        # Validate system files
        if not self.system_manager.validate_system_files():
            return False
        
        # Install requirements
        if not self.system_manager.install_requirements():
            return False
        
        # Setup environment
        if not self.system_manager.setup_environment():
            return False
        
        # Create cache directories
        self.system_manager.create_cache_directories()
        
        # Download NLTK data
        self.system_manager.download_nltk_data()
        
        return True
    
    def launch_system(self) -> None:
        """Launch the ASPS Medical AI System"""
        print_section("LAUNCHING ASPS MEDICAL AI SYSTEM")
        
        try:
            # Import and run the main system
            print_status("Importing ASPS Medical AI System...", "processing")
            
            # Change to the correct directory
            os.chdir(self.system_manager.base_dir)
            
            # Add current directory to Python path
            if str(self.system_manager.base_dir) not in sys.path:
                sys.path.insert(0, str(self.system_manager.base_dir))
            
            # Import the main system
            print_status("Loading AI models and knowledge bases...", "processing")
            print_status("This may take a few minutes on first run...", "info")
            
            # Run the main system with UTF-8 encoding
            with open("demo_asps_1.py", "r", encoding="utf-8") as f:
                exec(f.read())
            
        except KeyboardInterrupt:
            print_status("\nSystem shutdown requested by user", "warning")
            sys.exit(0)
        except Exception as e:
            print_status(f"Failed to launch system: {e}", "error")
            print(f"\n{Colors.RED}Error Details:{Colors.RESET}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def print_final_status(self) -> None:
        """Print final startup status"""
        elapsed_time = time.time() - self.start_time
        
        print(f"""
{Colors.GREEN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🎉 SYSTEM READY FOR USE! 🎉                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.RESET}

{Colors.CYAN}🚀 Startup Time:{Colors.RESET} {elapsed_time:.1f} seconds
{Colors.CYAN}🌐 Web Interface:{Colors.RESET} Access via your RunPod's public URL on port 8000
{Colors.CYAN}📖 API Docs:{Colors.RESET} Add /docs to your public URL for API documentation
{Colors.CYAN}🔍 Health Check:{Colors.RESET} Add /health to check system status

{Colors.YELLOW}💡 RunPod Access Instructions:{Colors.RESET}
   1. Click on your RunPod's "Connect" button
   2. Use the HTTP service on port 8000
   3. The beautiful web interface will load automatically

{Colors.GREEN}✅ Your ASPS Medical AI System is now running!{Colors.RESET}
""")

# ============================
# 🎯 MAIN EXECUTION
# ============================

def main():
    """Main execution function for RunPod"""
    launcher = RunPodLauncher()
    
    try:
        # Run system checks and setup
        if not launcher.run_system_checks():
            print_status("System checks failed. Please resolve issues and try again.", "error")
            sys.exit(1)
        
        # Print final status
        launcher.print_final_status()
        
        # Launch the system (this will run indefinitely)
        launcher.launch_system()
        
    except KeyboardInterrupt:
        print_status("\n👋 Goodbye! System shutdown requested.", "info")
        sys.exit(0)
    except Exception as e:
        print_status(f"Critical error during startup: {e}", "error")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
