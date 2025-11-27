# 🚀 GitHub Upload Guide

Complete guide to upload your Neural TTS project to GitHub.

---

## ⚡ Quick Upload (Automated)

```bash
python upload_github.py
```

This script will:
1. ✅ Initialize git repository
2. ✅ Configure git (name/email)
3. ✅ Add all files
4. ✅ Create initial commit
5. ✅ Add GitHub remote
6. ✅ Push to GitHub

---

## 📋 Manual Upload (Step-by-Step)

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `neural-tts` (or your choice)
3. Description: "Production-grade neural TTS with extreme clarity, voice cloning, and multilingual support"
4. **Important:** Do NOT initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

### Step 2: Initialize Git (If Not Already Done)

```bash
cd e:\tts
git init
```

### Step 3: Configure Git

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### Step 4: Add All Files

```bash
git add .
```

### Step 5: Create Initial Commit

```bash
git commit -m "Initial commit - Neural TTS v1.0.0"
```

### Step 6: Add Remote Repository

```bash
# Replace with your actual repository URL
git remote add origin https://github.com/yourusername/neural-tts.git
```

### Step 7: Push to GitHub

```bash
git branch -M main
git push -u origin main
```

---

## 🔑 Authentication

### Option 1: Personal Access Token (Recommended)

1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "Neural TTS Upload"
4. Select scopes: `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)

When pushing, use:
- Username: your GitHub username
- Password: the token you just copied

### Option 2: SSH Key

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: https://github.com/settings/keys
```

Then use SSH URL:
```bash
git remote set-url origin git@github.com:yourusername/neural-tts.git
```

---

## 📝 Recommended Repository Settings

### Description
```
Production-grade neural TTS with extreme clarity, voice cloning, and multilingual support. FastSpeech2 + HiFi-GAN implementation in pure Python.
```

### Topics (Add these tags)
```
tts
text-to-speech
speech-synthesis
voice-cloning
neural-tts
fastspeech2
hifigan
pytorch
python
deep-learning
multilingual
voice-conversion
```

### About Section
- ✅ Website: https://pypi.org/project/neural-tts/
- ✅ Add topics (see above)

---

## 📄 Update README Badges

Add these badges to the top of your README.md:

```markdown
# Neural TTS

[![PyPI version](https://badge.fury.io/py/neural-tts.svg)](https://badge.fury.io/py/neural-tts)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/neural-tts)](https://pepy.tech/project/neural-tts)
```

---

## 🎯 After Upload

### 1. Verify Upload
Visit: `https://github.com/yourusername/neural-tts`

### 2. Update pyproject.toml URLs

Edit `pyproject.toml`:
```toml
[project.urls]
Homepage = "https://github.com/yourusername/neural-tts"
Documentation = "https://github.com/yourusername/neural-tts#readme"
Repository = "https://github.com/yourusername/neural-tts"
Issues = "https://github.com/yourusername/neural-tts/issues"
```

### 3. Create Release

1. Go to your repository
2. Click "Releases" → "Create a new release"
3. Tag: `v1.0.0`
4. Title: "Neural TTS v1.0.0 - Initial Release"
5. Description:
   ```markdown
   ## 🎉 Initial Release
   
   Production-grade neural TTS system with:
   - ✅ Extreme clarity (HiFi-GAN vocoder)
   - ✅ Voice cloning (5-10 seconds)
   - ✅ Multilingual support (all languages)
   - ✅ Real-time inference (~30-155ms)
   - ✅ Prosody control (pitch, speed, energy)
   
   ### Installation
   ```bash
   pip install neural-tts
   ```
   
   ### Quick Start
   ```python
   from neural_tts import speak
   speak("Hello, world!")
   ```
   
   See [README](https://github.com/yourusername/neural-tts#readme) for full documentation.
   ```

### 4. Enable GitHub Pages (Optional)

1. Go to Settings → Pages
2. Source: Deploy from a branch
3. Branch: main, folder: / (root)
4. Your docs will be at: `https://yourusername.github.io/neural-tts/`

---

## 🔄 Future Updates

When you make changes:

```bash
# 1. Make your changes
# 2. Stage changes
git add .

# 3. Commit
git commit -m "Description of changes"

# 4. Push
git push
```

---

## 🐛 Troubleshooting

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/yourusername/neural-tts.git
```

### Error: "failed to push some refs"
```bash
# If repository has content
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Error: "Authentication failed"
- Use Personal Access Token instead of password
- Or set up SSH key

### Large files warning
```bash
# If you have model checkpoints, use Git LFS
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes
```

---

## 📊 Repository Structure

After upload, your repository will have:

```
neural-tts/
├── .git/                    # Git metadata
├── .gitignore              # Ignored files
├── LICENSE                 # MIT license
├── README.md               # Project overview
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
├── setup_verify.py         # Setup verification
├── publish.py              # PyPI publishing
├── upload_github.py        # GitHub upload (this script)
├── config/                 # Configuration files
├── text/                   # Text processing
├── models/                 # Neural models
├── data/                   # Data processing
├── training/               # Training scripts
├── inference/              # Inference engines
├── utils/                  # Utilities
├── examples/               # Usage examples
└── docs/                   # Documentation
    ├── ARCHITECTURE.md
    ├── TRAINING_GUIDE.md
    ├── INFERENCE_GUIDE.md
    └── ...
```

---

## ✨ Recommended Next Steps

1. ✅ Upload to GitHub (you're here!)
2. ⏭️ Add badges to README
3. ⏭️ Create first release (v1.0.0)
4. ⏭️ Set up GitHub Actions (CI/CD)
5. ⏭️ Add contributing guidelines
6. ⏭️ Create issue templates
7. ⏭️ Enable discussions

---

## 🎉 You're Done!

Your project will be available at:
- **GitHub:** `https://github.com/yourusername/neural-tts`
- **PyPI:** `https://pypi.org/project/neural-tts/`

Users can:
- **Install:** `pip install neural-tts`
- **Clone:** `git clone https://github.com/yourusername/neural-tts.git`
- **Contribute:** Fork and submit PRs

---

**Happy coding! 🚀**
