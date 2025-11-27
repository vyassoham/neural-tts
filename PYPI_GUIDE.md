# 🚀 Publishing to PyPI - Complete Guide

## One-Line Publishing (After Setup)

```bash
python -m build && twine upload dist/*
```

---

## 📦 Complete Publishing Guide

### Step 1: Prepare Your Package

1. **Update package information** in `pyproject.toml`:
   ```toml
   name = "neural-tts"  # Change if name is taken
   version = "1.0.0"
   authors = [{name = "Your Name", email = "your@email.com"}]
   ```

2. **Update URLs** in `pyproject.toml`:
   ```toml
   [project.urls]
   Homepage = "https://github.com/yourusername/neural-tts"
   Repository = "https://github.com/yourusername/neural-tts"
   ```

3. **Rename package directory**:
   ```bash
   # The package name should match pyproject.toml
   # Current: tts/
   # Should be: neural_tts/
   
   # On Windows:
   cd e:\
   ren tts neural_tts
   ```

### Step 2: Install Build Tools

```bash
pip install build twine
```

### Step 3: Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Create account and verify email
3. Enable 2FA (recommended)
4. Create API token: https://pypi.org/manage/account/token/
5. Save token securely

### Step 4: Build the Package

```bash
# Navigate to package directory
cd e:\neural_tts

# Build distribution
python -m build
```

This creates:
- `dist/neural_tts-1.0.0-py3-none-any.whl`
- `dist/neural_tts-1.0.0.tar.gz`

### Step 5: Upload to PyPI

```bash
# Upload to PyPI
twine upload dist/*

# You'll be prompted for:
# Username: __token__
# Password: <your-api-token>
```

### Step 6: Verify Installation

```bash
# Install from PyPI
pip install neural-tts

# Test it
python -c "from neural_tts import speak; print('Success!')"
```

---

## 🎯 One-Line Publishing (Complete)

After initial setup, future releases are just:

```bash
# Update version in pyproject.toml, then:
rm -rf dist/ build/ *.egg-info && python -m build && twine upload dist/*
```

Or on Windows:
```powershell
Remove-Item -Recurse -Force dist, build, *.egg-info; python -m build; twine upload dist/*
```

---

## 📱 Daily Usage After Publishing

### Method 1: Simple One-Liner

```python
from neural_tts import speak

# That's it! One line to speak:
speak("Hello, world!")
```

### Method 2: Python API

```python
from neural_tts import TTS

# Initialize once
tts = TTS()

# Use many times
tts.speak("Hello, world!")
tts.speak("How are you?")
tts.speak("This is amazing!", pitch_scale=1.2)
```

### Method 3: Command Line

```bash
# After pip install neural-tts

# Basic usage
neural-tts "Hello, world!" -o output.wav

# With options
neural-tts "Hello!" -o output.wav --pitch 1.2 --speed 0.9

# Different language
neural-tts "Hola, mundo!" -o output.wav --language es

# Voice cloning
neural-tts "Clone this" -o output.wav --clone reference.wav
```

### Method 4: In Scripts

```python
#!/usr/bin/env python3
from neural_tts import TTS

tts = TTS()

# Read from file
with open('book.txt') as f:
    text = f.read()
    tts.speak(text, output_file='audiobook.wav')

# Interactive
while True:
    text = input("Enter text (or 'quit'): ")
    if text == 'quit':
        break
    tts.speak(text, output_file='output.wav')
    print("✓ Saved to output.wav")
```

### Method 5: Web API

```python
from flask import Flask, request, send_file
from neural_tts import TTS

app = Flask(__name__)
tts = TTS()

@app.route('/speak', methods=['POST'])
def speak():
    text = request.json['text']
    tts.speak(text, output_file='temp.wav')
    return send_file('temp.wav', mimetype='audio/wav')

if __name__ == '__main__':
    app.run(port=5000)
```

---

## 🔧 Common Daily Use Cases

### 1. Quick Voice Notes

```python
from neural_tts import speak

# Record thoughts instantly
speak("Remember to buy milk")
speak("Meeting at 3 PM tomorrow")
```

### 2. Audiobook Creation

```python
from neural_tts import TTS
tts = TTS()

chapters = [
    "Chapter 1: The Beginning...",
    "Chapter 2: The Journey...",
    "Chapter 3: The End..."
]

for i, chapter in enumerate(chapters):
    tts.speak(chapter, output_file=f'chapter_{i+1}.wav')
```

### 3. Language Learning

```python
from neural_tts import TTS
tts = TTS()

phrases = {
    'en': "Hello, how are you?",
    'es': "Hola, ¿cómo estás?",
    'fr': "Bonjour, comment allez-vous?",
    'de': "Hallo, wie geht es dir?"
}

for lang, text in phrases.items():
    tts.speak(text, language=lang, output_file=f'phrase_{lang}.wav')
```

### 4. Accessibility Tool

```python
from neural_tts import TTS
import pyperclip  # pip install pyperclip

tts = TTS()

# Read clipboard content
text = pyperclip.paste()
tts.speak(text, output_file='clipboard.wav')
```

### 5. Podcast Generation

```python
from neural_tts import TTS
tts = TTS()

script = """
Welcome to our podcast!
Today we'll discuss neural networks.
Let's get started...
"""

tts.speak(script, output_file='podcast.wav', speed_scale=0.95)
```

---

## 🎨 Advanced Daily Usage

### Custom Voice Presets

```python
from neural_tts import TTS

class MyTTS:
    def __init__(self):
        self.tts = TTS()
    
    def happy(self, text):
        return self.tts.speak(text, pitch_scale=1.15, speed_scale=1.1, energy_scale=1.3)
    
    def sad(self, text):
        return self.tts.speak(text, pitch_scale=0.85, speed_scale=0.85, energy_scale=0.8)
    
    def excited(self, text):
        return self.tts.speak(text, pitch_scale=1.2, speed_scale=1.2, energy_scale=1.4)

tts = MyTTS()
tts.happy("I'm so happy!")
tts.sad("I'm feeling sad.")
tts.excited("This is amazing!")
```

### Batch Processing

```python
from neural_tts import TTS
from pathlib import Path

tts = TTS()

# Process all text files in a directory
for txt_file in Path('texts/').glob('*.txt'):
    with open(txt_file) as f:
        text = f.read()
    
    output = f'audio/{txt_file.stem}.wav'
    tts.speak(text, output_file=output)
    print(f"✓ {txt_file.name} → {output}")
```

### Real-time Streaming

```python
from neural_tts import TTS
import sounddevice as sd  # pip install sounddevice

tts = TTS()

def speak_now(text):
    """Speak immediately without saving to file"""
    audio = tts.speak(text)  # No output_file = returns array
    sd.play(audio, samplerate=22050)
    sd.wait()

speak_now("This plays immediately!")
```

---

## 📋 Installation for End Users

After publishing to PyPI, users can install with:

```bash
# Basic installation
pip install neural-tts

# With optional dependencies
pip install neural-tts[phonemizer]  # Better phonemization
pip install neural-tts[onnx]        # Faster inference
pip install neural-tts[dev]         # Development tools
```

---

## 🔄 Updating Your Package

1. **Make changes** to your code
2. **Update version** in `pyproject.toml`:
   ```toml
   version = "1.0.1"  # Increment version
   ```
3. **Rebuild and upload**:
   ```bash
   rm -rf dist/ build/ *.egg-info
   python -m build
   twine upload dist/*
   ```

---

## 🎯 Quick Reference

### Publishing Checklist

- [ ] Update `pyproject.toml` (name, version, author, URLs)
- [ ] Rename directory to match package name
- [ ] Install build tools: `pip install build twine`
- [ ] Create PyPI account and API token
- [ ] Build: `python -m build`
- [ ] Upload: `twine upload dist/*`
- [ ] Test: `pip install neural-tts`

### Daily Usage Checklist

- [ ] Install: `pip install neural-tts`
- [ ] Import: `from neural_tts import speak` or `from neural_tts import TTS`
- [ ] Use: `speak("Hello!")` or `tts.speak("Hello!")`

---

## 🚀 You're Ready!

**To publish:**
```bash
python -m build && twine upload dist/*
```

**To use daily:**
```python
from neural_tts import speak
speak("Hello, world!")
```

**That's it! 🎉**
