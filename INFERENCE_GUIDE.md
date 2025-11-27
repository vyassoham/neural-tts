# Inference Guide

Complete guide for using the Neural TTS system for speech synthesis.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Synthesis](#basic-synthesis)
3. [Prosody Control](#prosody-control)
4. [Voice Cloning](#voice-cloning)
5. [Multilingual Synthesis](#multilingual-synthesis)
6. [Advanced Usage](#advanced-usage)
7. [API Reference](#api-reference)

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Download Pre-trained Models

If you don't want to train from scratch, download pre-trained models:

```bash
# Create checkpoints directory
mkdir -p checkpoints/acoustic
mkdir -p checkpoints/vocoder

# Download models (example URLs - replace with actual)
# wget https://example.com/acoustic_model.pt -O checkpoints/acoustic/acoustic_model_final.pt
# wget https://example.com/vocoder_model.pt -O checkpoints/vocoder/generator_final.pt
```

### First Synthesis

```python
from inference.synthesizer import TTSSynthesizer

# Initialize
tts = TTSSynthesizer(
    acoustic_checkpoint='checkpoints/acoustic/acoustic_model_final.pt',
    vocoder_checkpoint='checkpoints/vocoder/generator_final.pt'
)

# Synthesize
audio = tts.synthesize("Hello, world!", language='en')

# Save
tts.save_audio(audio, 'output.wav')
```

---

## Basic Synthesis

### Simple Text-to-Speech

```python
from inference.synthesizer import TTSSynthesizer

# Initialize synthesizer
tts = TTSSynthesizer(
    acoustic_checkpoint='checkpoints/acoustic/acoustic_model_final.pt',
    vocoder_checkpoint='checkpoints/vocoder/generator_final.pt'
)

# Synthesize speech
text = "This is a test of the text to speech system."
audio = tts.synthesize(text, language='en')

# Save to file
tts.save_audio(audio, 'outputs/test.wav')
```

### Batch Processing

Process multiple texts efficiently:

```python
texts = [
    "First sentence to synthesize.",
    "Second sentence to synthesize.",
    "Third sentence to synthesize."
]

# Synthesize all at once
audios = tts.synthesize_batch(texts, language='en')

# Save each audio
for i, audio in enumerate(audios):
    tts.save_audio(audio, f'outputs/batch_{i+1}.wav')
```

### Long Text Synthesis

For long texts, split into sentences:

```python
long_text = """
This is a very long text that needs to be synthesized.
It contains multiple sentences and paragraphs.
The system will handle it efficiently.
"""

# Synthesize
audio = tts.synthesize(long_text, language='en')
tts.save_audio(audio, 'outputs/long_text.wav')
```

---

## Prosody Control

### Pitch Control

Adjust the pitch of the synthesized speech:

```python
# Normal pitch
audio = tts.synthesize("Hello", language='en', pitch_scale=1.0)

# Higher pitch (1.2x)
audio_high = tts.synthesize("Hello", language='en', pitch_scale=1.2)

# Lower pitch (0.8x)
audio_low = tts.synthesize("Hello", language='en', pitch_scale=0.8)
```

**Recommended ranges:**
- `0.5` - Very low (deep voice)
- `0.8` - Slightly low
- `1.0` - Normal
- `1.2` - Slightly high
- `1.5` - Very high (chipmunk effect)

### Speed Control

Adjust speaking rate:

```python
# Normal speed
audio = tts.synthesize("Hello", language='en', speed_scale=1.0)

# Slower (0.8x speed = 1.25x duration)
audio_slow = tts.synthesize("Hello", language='en', speed_scale=0.8)

# Faster (1.3x speed = 0.77x duration)
audio_fast = tts.synthesize("Hello", language='en', speed_scale=1.3)
```

**Recommended ranges:**
- `0.5` - Very slow
- `0.8` - Slightly slow
- `1.0` - Normal
- `1.3` - Slightly fast
- `2.0` - Very fast

### Energy Control

Adjust volume/intensity:

```python
# Normal energy
audio = tts.synthesize("Hello", language='en', energy_scale=1.0)

# Louder
audio_loud = tts.synthesize("Hello", language='en', energy_scale=1.5)

# Quieter
audio_quiet = tts.synthesize("Hello", language='en', energy_scale=0.7)
```

### Combined Control

Combine multiple prosody controls:

```python
audio = tts.synthesize(
    "This is expressive speech!",
    language='en',
    pitch_scale=1.1,    # Slightly higher pitch
    speed_scale=0.9,    # Slightly slower
    energy_scale=1.2    # Slightly louder
)
```

### Emotion Presets

Create emotion presets:

```python
# Happy voice
def synthesize_happy(text):
    return tts.synthesize(
        text,
        language='en',
        pitch_scale=1.15,
        speed_scale=1.1,
        energy_scale=1.3
    )

# Sad voice
def synthesize_sad(text):
    return tts.synthesize(
        text,
        language='en',
        pitch_scale=0.85,
        speed_scale=0.85,
        energy_scale=0.8
    )

# Excited voice
def synthesize_excited(text):
    return tts.synthesize(
        text,
        language='en',
        pitch_scale=1.2,
        speed_scale=1.2,
        energy_scale=1.4
    )

# Use presets
audio_happy = synthesize_happy("I'm so happy!")
audio_sad = synthesize_sad("I'm feeling sad.")
audio_excited = synthesize_excited("This is amazing!")
```

---

## Voice Cloning

### Basic Voice Cloning

Clone a voice from reference audio:

```python
from inference.voice_cloner import VoiceCloner

# Initialize cloner
cloner = VoiceCloner(
    acoustic_checkpoint='checkpoints/acoustic/acoustic_model_final.pt',
    vocoder_checkpoint='checkpoints/vocoder/generator_final.pt',
    speaker_encoder_checkpoint='checkpoints/speaker/encoder_final.pt'
)

# Clone voice
reference_audio = 'reference_voice.wav'  # 5-10 seconds of clean speech
text = "This will be spoken in the cloned voice."

audio = cloner.clone_voice(text, reference_audio, language='en')
cloner.save_audio(audio, 'outputs/cloned.wav')
```

### Reusing Speaker Embeddings

Extract embedding once, use multiple times:

```python
# Extract speaker embedding
speaker_embedding = cloner.extract_speaker_embedding('reference_voice.wav')

# Use for multiple texts
texts = [
    "First sentence in cloned voice.",
    "Second sentence in cloned voice.",
    "Third sentence in cloned voice."
]

for i, text in enumerate(texts):
    audio = cloner.clone_voice_from_embedding(
        text,
        speaker_embedding,
        language='en'
    )
    cloner.save_audio(audio, f'outputs/cloned_{i+1}.wav')
```

### Voice Similarity Comparison

Compare speaker similarity:

```python
# Compare two audio files
similarity = cloner.compare_speakers('voice1.wav', 'voice2.wav')
print(f"Similarity: {similarity:.4f}")  # 0.0 to 1.0

# Typical values:
# > 0.9: Same speaker
# 0.7-0.9: Similar voices
# 0.5-0.7: Different but related
# < 0.5: Very different speakers
```

---

## Multilingual Synthesis

### Supported Languages

The system supports all languages through Unicode normalization:

```python
# English
audio_en = tts.synthesize("Hello, how are you?", language='en')

# Spanish
audio_es = tts.synthesize("Hola, ¿cómo estás?", language='es')

# French
audio_fr = tts.synthesize("Bonjour, comment allez-vous?", language='fr')

# German
audio_de = tts.synthesize("Hallo, wie geht es dir?", language='de')

# Chinese
audio_zh = tts.synthesize("你好，你好吗？", language='zh')

# Japanese
audio_ja = tts.synthesize("こんにちは、お元気ですか？", language='ja')

# Korean
audio_ko = tts.synthesize("안녕하세요, 어떻게 지내세요?", language='ko')

# Russian
audio_ru = tts.synthesize("Здравствуйте, как дела?", language='ru')

# Arabic
audio_ar = tts.synthesize("مرحبا، كيف حالك؟", language='ar')

# Hindi
audio_hi = tts.synthesize("नमस्ते, आप कैसे हैं?", language='hi')
```

### Language Codes

Common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `ru` - Russian
- `ar` - Arabic
- `hi` - Hindi
- `tr` - Turkish
- `pl` - Polish
- `nl` - Dutch
- `sv` - Swedish

---

## Advanced Usage

### Custom Sampling Rate

```python
# Save with different sampling rate
audio = tts.synthesize("Hello", language='en')
tts.save_audio(audio, 'output_44k.wav', sr=44100)
```

### Mel-Spectrogram Generation

Generate mel-spectrogram without vocoding:

```python
# Get mel-spectrogram
mel = tts.text_to_mel("Hello, world!", language='en')

# mel shape: (n_mel_channels, time)
print(f"Mel shape: {mel.shape}")

# Visualize
from utils.tools import plot_spectrogram
plot_spectrogram(mel, save_path='outputs/mel.png')
```

### Custom Vocoding

Use custom mel-spectrogram:

```python
import torch

# Load or generate mel-spectrogram
mel = torch.randn(80, 100)  # (n_mel, time)

# Convert to audio
audio = tts.mel_to_audio(mel)
tts.save_audio(audio, 'outputs/custom.wav')
```

### Multi-Speaker Synthesis

For multi-speaker models:

```python
# Synthesize with different speakers
audio_speaker0 = tts.synthesize("Hello", language='en', speaker_id=0)
audio_speaker1 = tts.synthesize("Hello", language='en', speaker_id=1)
audio_speaker2 = tts.synthesize("Hello", language='en', speaker_id=2)
```

### Streaming Synthesis

For real-time applications:

```python
# Process in chunks
def synthesize_streaming(text, chunk_size=50):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) 
              for i in range(0, len(words), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        audio = tts.synthesize(chunk, language='en')
        tts.save_audio(audio, f'outputs/chunk_{i}.wav')
        # Or stream directly to audio output
```

---

## API Reference

### TTSSynthesizer

```python
class TTSSynthesizer:
    def __init__(self, acoustic_checkpoint, vocoder_checkpoint, 
                 acoustic_config=None, vocoder_config=None, device=None)
    
    def synthesize(self, text, language='en', speaker_id=0,
                   pitch_scale=1.0, speed_scale=1.0, energy_scale=1.0)
    
    def synthesize_batch(self, texts, language='en', speaker_ids=None,
                         pitch_scales=None, speed_scales=None, energy_scales=None)
    
    def save_audio(self, audio, path, sr=None)
    
    def text_to_mel(self, text, language='en', speaker_id=0,
                    pitch_scale=1.0, speed_scale=1.0, energy_scale=1.0)
    
    def mel_to_audio(self, mel)
```

### VoiceCloner

```python
class VoiceCloner:
    def __init__(self, acoustic_checkpoint, vocoder_checkpoint,
                 speaker_encoder_checkpoint, acoustic_config=None,
                 vocoder_config=None, device=None)
    
    def extract_speaker_embedding(self, audio_path)
    
    def clone_voice(self, text, reference_audio, language='en',
                    pitch_scale=1.0, speed_scale=1.0, energy_scale=1.0)
    
    def clone_voice_from_embedding(self, text, speaker_embedding, language='en',
                                    pitch_scale=1.0, speed_scale=1.0, energy_scale=1.0)
    
    def save_audio(self, audio, path, sr=None)
    
    def compare_speakers(self, audio_path1, audio_path2)
```

---

## Performance Tips

### Speed Optimization

1. **Use GPU:**
   ```python
   tts = TTSSynthesizer(..., device='cuda')
   ```

2. **Batch Processing:**
   ```python
   # Faster than individual calls
   audios = tts.synthesize_batch(texts)
   ```

3. **Reuse Embeddings:**
   ```python
   # Extract once, use many times
   embedding = cloner.extract_speaker_embedding(ref_audio)
   ```

### Quality Optimization

1. **Use Higher Sampling Rate:**
   ```python
   # In config/vocoder_config.py
   sampling_rate = 44100
   ```

2. **Adjust Prosody:**
   ```python
   # Fine-tune for natural speech
   audio = tts.synthesize(text, pitch_scale=1.05, speed_scale=0.95)
   ```

3. **Use Better Models:**
   - Train longer
   - Use more data
   - Increase model size

---

## Troubleshooting

### Audio Quality Issues

**Problem:** Robotic or unnatural voice
**Solution:**
- Train vocoder longer
- Use better training data
- Adjust prosody controls

**Problem:** Muffled audio
**Solution:**
- Check sampling rate
- Verify audio preprocessing
- Increase model size

### Performance Issues

**Problem:** Slow synthesis
**Solution:**
- Use GPU
- Reduce model size
- Export to ONNX

**Problem:** Out of memory
**Solution:**
- Use CPU
- Reduce batch size
- Process in chunks

---

## Examples

See the `examples/` directory for complete examples:

- `example_synthesis.py` - Basic synthesis with prosody control
- `example_voice_cloning.py` - Voice cloning examples
- `example_multilingual.py` - Multilingual synthesis

Run examples:

```bash
python examples/example_synthesis.py
python examples/example_voice_cloning.py
python examples/example_multilingual.py
```

---

**Happy synthesizing! 🎤**
