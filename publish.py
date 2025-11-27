#!/usr/bin/env python3
"""
Quick publish script for PyPI
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    
    print(f"✅ Success!")
    if result.stdout:
        print(result.stdout)
    
    return True


def main():
    """Main publish workflow"""
    print("\n" + "="*60)
    print("Neural TTS - PyPI Publishing Script")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path('pyproject.toml').exists():
        print("❌ Error: pyproject.toml not found!")
        print("Please run this script from the package root directory.")
        sys.exit(1)
    
    # Step 1: Clean previous builds
    print("\n📦 Step 1: Cleaning previous builds...")
    dirs_to_remove = ['dist', 'build', '*.egg-info']
    for pattern in dirs_to_remove:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  Removed: {path}")
    
    # Step 2: Check build tools
    print("\n🔧 Step 2: Checking build tools...")
    try:
        import build
        import twine
        print("  ✅ build and twine are installed")
    except ImportError:
        print("  ❌ Missing build tools!")
        print("\n  Installing build tools...")
        if not run_command("pip install build twine", "Installing build and twine"):
            sys.exit(1)
    
    # Step 3: Build the package
    if not run_command("python -m build", "Step 3: Building package"):
        sys.exit(1)
    
    # Step 4: Check the distribution
    if not run_command("twine check dist/*", "Step 4: Checking distribution"):
        sys.exit(1)
    
    # Step 5: Upload to PyPI
    print("\n" + "="*60)
    print("Step 5: Upload to PyPI")
    print("="*60)
    print("\nOptions:")
    print("1. Upload to PyPI (production)")
    print("2. Upload to TestPyPI (testing)")
    print("3. Skip upload")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        print("\n📤 Uploading to PyPI...")
        print("\nYou'll need your PyPI API token.")
        print("Username: __token__")
        print("Password: <your-api-token>")
        
        if not run_command("twine upload dist/*", "Uploading to PyPI"):
            sys.exit(1)
        
        print("\n" + "="*60)
        print("🎉 Successfully published to PyPI!")
        print("="*60)
        print("\nUsers can now install with:")
        print("  pip install neural-tts")
        
    elif choice == '2':
        print("\n📤 Uploading to TestPyPI...")
        
        if not run_command(
            "twine upload --repository testpypi dist/*",
            "Uploading to TestPyPI"
        ):
            sys.exit(1)
        
        print("\n" + "="*60)
        print("🎉 Successfully published to TestPyPI!")
        print("="*60)
        print("\nTest installation with:")
        print("  pip install --index-url https://test.pypi.org/simple/ neural-tts")
        
    else:
        print("\n⏭️  Skipping upload.")
        print("\nTo upload manually:")
        print("  twine upload dist/*")
    
    print("\n" + "="*60)
    print("✅ Build complete!")
    print("="*60)
    print("\nGenerated files:")
    for file in Path('dist').glob('*'):
        print(f"  - {file}")


if __name__ == '__main__':
    main()
