#!/usr/bin/env python3
"""
LQMF Fine-Tuning Dependencies Installer

This script installs all required dependencies for the fine-tuning system,
including PEFT, BitsAndBytes, and other necessary packages.
"""

import subprocess
import sys
import importlib
from pathlib import Path

def run_command(cmd):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_package(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package):
    """Install a package using pip."""
    print(f"Installing {package}...")
    success, output = run_command(f"{sys.executable} -m pip install {package}")
    if success:
        print(f"✅ {package} installed successfully")
        return True
    else:
        print(f"❌ Failed to install {package}: {output}")
        return False

def main():
    print("🔧 LQMF Fine-Tuning Dependencies Installer")
    print("=" * 50)
    
    # Required packages for fine-tuning
    required_packages = [
        ("peft", "peft>=0.7.0"),
        ("bitsandbytes", "bitsandbytes>=0.41.0"),
        ("accelerate", "accelerate>=0.24.0"),
        ("datasets", "datasets>=2.14.0"),
        ("transformers", "transformers>=4.35.0"),
        ("torch", "torch>=2.0.0"),
        ("nltk", "nltk"),
        ("rouge_score", "rouge-score"),
        ("psutil", "psutil"),
    ]
    
    # Check current installations
    print("\n📋 Checking current installations:")
    missing_packages = []
    
    for package_name, pip_name in required_packages:
        if check_package(package_name):
            print(f"✅ {package_name} is already installed")
        else:
            print(f"❌ {package_name} is missing")
            missing_packages.append(pip_name)
    
    if not missing_packages:
        print("\n🎉 All required packages are already installed!")
        return
    
    print(f"\n📦 Installing {len(missing_packages)} missing packages...")
    
    # Install missing packages
    failed_packages = []
    for package in missing_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print("\n" + "=" * 50)
    
    if failed_packages:
        print("⚠️  Some packages failed to install:")
        for package in failed_packages:
            print(f"   - {package}")
        print("\nTry installing manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
    else:
        print("🎉 All dependencies installed successfully!")
        print("\n🚀 Fine-tuning system is ready to use!")
        print("   Run: python cli/finetuning_cli.py")
    
    # Download NLTK data if needed
    try:
        import nltk
        print("\n📥 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        print("✅ NLTK data downloaded")
    except:
        print("⚠️  NLTK data download failed (optional)")

if __name__ == "__main__":
    main()