# 🎉 PyPI Publishing & Daily Usage - Complete Summary

## ⚡ ULTRA-QUICK ANSWER

### To Publish (One Line):
```bash
python publish.py
```

### To Use Daily (One Line):
```python
from neural_tts import speak; speak("Hello, world!")
```

---

## 📦 What I Just Added

I created **8 new files** to make publishing and daily usage super easy:

1. **`__init__.py`** - Simplified API with `speak()` function
2. **`pyproject.toml`** - Modern PyPI package configuration
3. **`cli.py`** - Command-line interface
4. **`LICENSE`** - MIT license
5. **`.gitignore`** - Git ignore file
6. **`publish.py`** - Automated publishing script
7. **`PYPI_GUIDE.md`** - Complete publishing guide
8. **`QUICK_START.md`** - Quick reference

---

## 🚀 Publishing to PyPI

### Option 1: Automated (Recommended)

```bash
# Just run this:
python publish.py

# Follow the prompts:
# 1. It cleans old builds
# 2. Installs build tools if needed
# 3. Builds the package
# 4. Asks if you want to upload to PyPI
```

### Option 2: Manual One-Liner

```bash
python -m build && twine upload dist/*
```

### First-Time Setup (One Time Only)

1. **Install build tools:**
   ```bash
   pip install build twine
   ```

2. **Create PyPI account:**
   - Go to https://pypi.org/account/register/
   - Create account and verify email
   - Create API token at https://pypi.org/manage/account/token/

3. **Update package info** in `pyproject.toml`:
   ```toml
   name = "neural-tts"  # Change if taken
   version = "1.0.0"
   authors = [{name = "Your Name", email = "your@email.com"}]
   ```

4. **Publish:**
   ```bash
   python publish.py
   ```

---

## 📱 Daily Usage (After Publishing)

### Installation

```bash
pip install neural-tts
```

### Usage Examples

#### 1. Simplest Possible (One Line!)

```python
from neural_tts import speak

speak("Hello, world!")
```

#### 2. Python API

```python
from neural_tts import TTS

# Initialize once
tts = TTS()

# Use many times
tts.speak("Hello, world!")
tts.speak("How are you?")
tts.speak("This is great!", pitch_scale=1.2)
```

#### 3. Command Line

```bash
# After pip install neural-tts

# Basic
neural-tts "Hello, world!" -o output.wav

# With options
neural-tts "Hello!" --pitch 1.2 --speed 0.9 -o output.wav

# Different language
neural-tts "Hola!" --language es -o output.wav
```

#### 4. In Your Scripts

```python
from neural_tts import TTS

tts = TTS()

# Read file
with open('article.txt') as f:
    tts.speak(f.read(), output_file='article.wav')

# Multiple languages
for lang, text in [('en', 'Hello'), ('es', 'Hola'), ('fr', 'Bonjour')]:
    tts.speak(text, language=lang, output_file=f'{lang}.wav')

# Batch processing
texts = ["First", "Second", "Third"]
for i, text in enumerate(texts):
    tts.speak(text, output_file=f'output_{i}.wav')
```

#### 5. Real-World Examples

**Audiobook Creator:**
```python
from neural_tts import TTS
tts = TTS()

with open('book.txt') as f:
    chapters = f.read().split('\n\n')

for i, chapter in enumerate(chapters):
    tts.speak(chapter, output_file=f'chapter_{i+1}.wav')
```

**Voice Assistant:**
```python
from neural_tts import TTS
tts = TTS()

def assistant_speak(text):
    tts.speak(text, output_file='response.wav')
    # Play audio here

assistant_speak("How can I help you?")
```

**Language Learning:**
```python
from neural_tts import TTS
tts = TTS()

phrases = {
    'en': "Good morning",
    'es': "Buenos días",
    'fr': "Bonjour",
    'de': "Guten Morgen"
}

for lang, phrase in phrases.items():
    tts.speak(phrase, language=lang, output_file=f'morning_{lang}.wav')
```

---

## 🎨 Advanced Daily Usage

### Custom Voice Presets

```python
from neural_tts import TTS

class VoicePresets:
    def __init__(self):
        self.tts = TTS()
    
    def happy(self, text):
        return self.tts.speak(text, pitch_scale=1.15, speed_scale=1.1, energy_scale=1.3)
    
    def sad(self, text):
        return self.tts.speak(text, pitch_scale=0.85, speed_scale=0.85, energy_scale=0.8)
    
    def excited(self, text):
        return self.tts.speak(text, pitch_scale=1.2, speed_scale=1.2, energy_scale=1.4)
    
    def calm(self, text):
        return self.tts.speak(text, pitch_scale=0.95, speed_scale=0.85, energy_scale=0.9)

voice = VoicePresets()
voice.happy("I'm so happy!")
voice.sad("I'm feeling sad.")
voice.excited("This is amazing!")
voice.calm("Relax and breathe.")
```

### Web API

```python
from flask import Flask, request, send_file
from neural_tts import TTS

app = Flask(__name__)
tts = TTS()

@app.route('/speak', methods=['POST'])
def speak():
    data = request.json
    text = data.get('text', '')
    language = data.get('language', 'en')
    
    tts.speak(text, language=language, output_file='temp.wav')
    return send_file('temp.wav', mimetype='audio/wav')

if __name__ == '__main__':
    app.run(port=5000)

# Usage:
# curl -X POST http://localhost:5000/speak \
#   -H "Content-Type: application/json" \
#   -d '{"text": "Hello, world!", "language": "en"}' \
#   --output output.wav
```

---

## 📊 Comparison: Before vs After

### Before (Complex)
```python
import sys
sys.path.append('e:/tts')
from inference.synthesizer import TTSSynthesizer

tts = TTSSynthesizer(
    acoustic_checkpoint='checkpoints/acoustic/acoustic_model_final.pt',
    vocoder_checkpoint='checkpoints/vocoder/generator_final.pt'
)

audio = tts.synthesize("Hello, world!", language='en')
tts.save_audio(audio, 'output.wav')
```

### After (Simple)
```python
from neural_tts import speak

speak("Hello, world!")
```

**That's 10 lines → 1 line! 🎉**

---

## 🔄 Publishing Updates

When you make changes and want to publish a new version:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "1.0.1"  # Increment
   ```

2. **Publish:**
   ```bash
   python publish.py
   ```

That's it! Users can update with:
```bash
pip install --upgrade neural-tts
```

---

## 📋 Complete Workflow

### For Package Maintainer (You)

```bash
# 1. Make changes to code
# 2. Update version in pyproject.toml
# 3. Publish
python publish.py

# Done! ✅
```

### For End Users

```bash
# 1. Install
pip install neural-tts

# 2. Use
python -c "from neural_tts import speak; speak('Hello!')"

# Done! ✅
```

---

## 🎯 Key Files Created

| File | Purpose |
|------|---------|
| `__init__.py` | Simplified API with `speak()` function |
| `pyproject.toml` | PyPI package configuration |
| `cli.py` | Command-line interface |
| `publish.py` | Automated publishing script |
| `LICENSE` | MIT license |
| `.gitignore` | Git ignore rules |
| `PYPI_GUIDE.md` | Detailed publishing guide |
| `QUICK_START.md` | Quick reference |

---

## 🌟 Benefits of PyPI Publishing

### Before Publishing
- ❌ Complex imports
- ❌ Manual path setup
- ❌ Hard to share
- ❌ No version control
- ❌ Manual updates

### After Publishing
- ✅ Simple `pip install`
- ✅ One-line usage
- ✅ Easy sharing
- ✅ Version management
- ✅ Automatic updates

---

## 🎓 Learning Path

### Beginner
1. Install: `pip install neural-tts`
2. Use: `from neural_tts import speak; speak("Hello!")`
3. Done!

### Intermediate
1. Use TTS class for more control
2. Adjust prosody (pitch, speed, energy)
3. Try different languages

### Advanced
1. Create custom voice presets
2. Build web APIs
3. Integrate into applications

---

## 💡 Pro Tips

### Tip 1: Cache TTS Instance
```python
# Initialize once (slow)
from neural_tts import TTS
tts = TTS()

# Use many times (fast)
for text in texts:
    tts.speak(text)
```

### Tip 2: Use Command Line for Quick Tests
```bash
neural-tts "Test this quickly" -o test.wav
```

### Tip 3: Create Aliases
```python
# In your project
from neural_tts import speak as say

say("Much shorter!")
```

### Tip 4: Environment Variables
```python
import os
os.environ['NEURAL_TTS_MODEL_PATH'] = '/path/to/models'

from neural_tts import TTS
tts = TTS()  # Auto-loads from env var
```

---

## 🚀 Summary

### To Publish (One Command):
```bash
python publish.py
```

### To Use Daily (One Line):
```python
from neural_tts import speak; speak("Hello, world!")
```

### That's It! 🎉

---

## 📚 Documentation

- **QUICK_START.md** - This file (quick reference)
- **PYPI_GUIDE.md** - Detailed publishing guide
- **README.md** - Project overview
- **ARCHITECTURE.md** - Technical details
- **TRAINING_GUIDE.md** - Training instructions
- **INFERENCE_GUIDE.md** - Usage guide

---

**You're all set! Publish once, use everywhere! 🚀**
