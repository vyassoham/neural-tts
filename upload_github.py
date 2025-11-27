#!/usr/bin/env python3
"""
GitHub Upload Script
Automates the process of uploading the Neural TTS project to GitHub
"""

import os
import subprocess
import sys


def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0 and check:
        print(f"❌ Error: {result.stderr}")
        return False
    
    print(f"✅ Success!")
    if result.stdout:
        print(result.stdout)
    
    return True


def check_git_installed():
    """Check if git is installed"""
    result = subprocess.run("git --version", shell=True, capture_output=True, text=True)
    return result.returncode == 0


def main():
    """Main GitHub upload workflow"""
    print("\n" + "="*60)
    print("Neural TTS - GitHub Upload Script")
    print("="*60)
    
    # Check if git is installed
    if not check_git_installed():
        print("\n❌ Git is not installed!")
        print("\nPlease install Git:")
        print("  Download from: https://git-scm.com/downloads")
        print("  Or use: winget install Git.Git")
        sys.exit(1)
    
    print("\n✅ Git is installed")
    
    # Check if already a git repository
    if os.path.exists('.git'):
        print("\n⚠️  This directory is already a git repository.")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Initialize git repository
    if not os.path.exists('.git'):
        if not run_command("git init", "Step 1: Initializing Git repository"):
            sys.exit(1)
    
    # Configure git (if not already configured)
    print("\n" + "="*60)
    print("Step 2: Git Configuration")
    print("="*60)
    
    # Check if user.name is set
    result = subprocess.run("git config user.name", shell=True, capture_output=True, text=True)
    if not result.stdout.strip():
        name = input("Enter your name: ")
        run_command(f'git config user.name "{name}"', "Setting user name", check=False)
    else:
        print(f"✅ User name: {result.stdout.strip()}")
    
    # Check if user.email is set
    result = subprocess.run("git config user.email", shell=True, capture_output=True, text=True)
    if not result.stdout.strip():
        email = input("Enter your email: ")
        run_command(f'git config user.email "{email}"', "Setting user email", check=False)
    else:
        print(f"✅ User email: {result.stdout.strip()}")
    
    # Add all files
    if not run_command("git add .", "Step 3: Adding all files"):
        sys.exit(1)
    
    # Commit
    commit_message = input("\nEnter commit message (default: 'Initial commit - Neural TTS v1.0.0'): ").strip()
    if not commit_message:
        commit_message = "Initial commit - Neural TTS v1.0.0"
    
    if not run_command(f'git commit -m "{commit_message}"', "Step 4: Creating initial commit"):
        print("\n⚠️  Note: If there are no changes to commit, this is normal.")
    
    # Get GitHub repository URL
    print("\n" + "="*60)
    print("Step 5: GitHub Repository Setup")
    print("="*60)
    print("\nYou need to create a GitHub repository first:")
    print("1. Go to https://github.com/new")
    print("2. Create a new repository (e.g., 'neural-tts')")
    print("3. Do NOT initialize with README, .gitignore, or license")
    print("4. Copy the repository URL")
    
    repo_url = input("\nEnter your GitHub repository URL (e.g., https://github.com/username/neural-tts.git): ").strip()
    
    if not repo_url:
        print("\n❌ No repository URL provided!")
        print("\nYou can add it later with:")
        print(f"  git remote add origin <your-repo-url>")
        print(f"  git branch -M main")
        print(f"  git push -u origin main")
        sys.exit(1)
    
    # Add remote
    # Check if remote already exists
    result = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"\n⚠️  Remote 'origin' already exists: {result.stdout.strip()}")
        response = input("Do you want to update it? (y/n): ")
        if response.lower() == 'y':
            run_command(f'git remote set-url origin {repo_url}', "Updating remote origin", check=False)
    else:
        if not run_command(f'git remote add origin {repo_url}', "Step 6: Adding remote repository"):
            sys.exit(1)
    
    # Rename branch to main
    run_command("git branch -M main", "Step 7: Renaming branch to main", check=False)
    
    # Push to GitHub
    print("\n" + "="*60)
    print("Step 8: Pushing to GitHub")
    print("="*60)
    print("\nYou may be prompted for your GitHub credentials.")
    print("If you have 2FA enabled, you'll need a Personal Access Token:")
    print("  Create one at: https://github.com/settings/tokens")
    
    response = input("\nReady to push? (y/n): ")
    if response.lower() != 'y':
        print("\nSkipping push. You can push later with:")
        print("  git push -u origin main")
        sys.exit(0)
    
    if not run_command("git push -u origin main", "Pushing to GitHub"):
        print("\n⚠️  Push failed. This might be due to:")
        print("  - Authentication issues (use Personal Access Token)")
        print("  - Repository already has content")
        print("  - Network issues")
        print("\nTry manually:")
        print("  git push -u origin main")
        sys.exit(1)
    
    # Success!
    print("\n" + "="*60)
    print("🎉 Successfully uploaded to GitHub!")
    print("="*60)
    print(f"\nYour repository: {repo_url.replace('.git', '')}")
    print("\nNext steps:")
    print("1. Visit your repository on GitHub")
    print("2. Add a description and topics")
    print("3. Enable GitHub Pages (optional)")
    print("4. Set up CI/CD (optional)")
    
    print("\n" + "="*60)
    print("✅ Upload complete!")
    print("="*60)


if __name__ == '__main__':
    main()
