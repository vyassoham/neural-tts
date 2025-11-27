# Neural TTS System - Production Grade

A complete, production-ready Text-to-Speech system built in pure Python with PyTorch.

## Features

- **Extreme Clarity**: HiFi-GAN vocoder for near-human audio quality
- **Multilingual**: Unicode-based phoneme processing for all languages
- **Fast**: Optimized FastSpeech2 architecture for real-time inference
- **Speaker Cloning**: Clone any voice with 5-10 seconds of audio
- **Flexible**: Adjustable pitch, speed, emotion, and tone
- **Multi-speaker**: Support for multiple voice identities
- **Low Latency**: CPU and GPU support with optimized inference
- **Modular**: Clean architecture for easy customization

## Architecture

### 1. Text Processor (`text/`)
- **Phonemizer**: Converts text to phonemes using language-specific rules
- **Normalizer**: Unicode normalization and text cleaning
- **Tokenizer**: Converts phonemes to model-ready tokens

### 2. Acoustic Model (`models/acoustic/`)
- **FastSpeech2**: Non-autoregressive transformer for mel-spectrogram generation
- **Variance Adaptors**: Pitch, energy, and duration predictors
- **Multi-head Attention**: Captures long-range dependencies

### 3. Vocoder (`models/vocoder/`)
- **HiFi-GAN**: High-fidelity generative adversarial network
- **Multi-scale discriminators**: Ensures audio quality at multiple resolutions
- **Multi-period discriminators**: Captures periodic patterns in speech

### 4. Speaker Encoder (`models/speaker/`)
- **Speaker Embedding**: Extracts voice characteristics from reference audio
- **Voice Cloning**: Adapts model to new speakers with minimal data

## Project Structure

```
tts/
├── config/
│   ├── __init__.py
│   ├── acoustic_config.py      # FastSpeech2 hyperparameters
│   ├── vocoder_config.py       # HiFi-GAN hyperparameters
│   └── training_config.py      # Training settings
├── text/
│   ├── __init__.py
│   ├── phonemizer.py           # Text to phoneme conversion
│   ├── normalizer.py           # Text normalization
│   └── symbols.py              # Phoneme symbols and mappings
├── models/
│   ├── __init__.py
│   ├── acoustic/
│   │   ├── __init__.py
│   │   ├── fastspeech2.py      # Main acoustic model
│   │   ├── transformer.py      # Transformer blocks
│   │   └── variance_adaptor.py # Pitch/duration/energy predictors
│   ├── vocoder/
│   │   ├── __init__.py
│   │   ├── hifigan.py          # HiFi-GAN generator
│   │   └── discriminator.py    # Multi-scale/period discriminators
│   └── speaker/
│       ├── __init__.py
│       └── encoder.py          # Speaker embedding network
├── data/
│   ├── __init__.py
│   ├── dataset.py              # PyTorch dataset classes
│   └── preprocessing.py        # Audio preprocessing utilities
├── training/
│   ├── __init__.py
│   ├── train_acoustic.py       # Train FastSpeech2
│   ├── train_vocoder.py        # Train HiFi-GAN
│   └── losses.py               # Loss functions
├── inference/
│   ├── __init__.py
│   ├── synthesizer.py          # Main TTS inference engine
│   └── voice_cloner.py         # Speaker cloning utilities
├── utils/
│   ├── __init__.py
│   ├── audio.py                # Audio processing utilities
│   └── tools.py                # General utilities
├── checkpoints/                # Model checkpoints (created during training)
├── outputs/                    # Generated audio files
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Inference (Pre-trained)

```python
from inference.synthesizer import TTSSynthesizer

# Initialize synthesizer
tts = TTSSynthesizer(
    acoustic_checkpoint='checkpoints/acoustic_model.pt',
    vocoder_checkpoint='checkpoints/vocoder_model.pt'
)

# Generate speech
audio = tts.synthesize(
    text="Hello, this is a test of the neural TTS system.",
    language="en",
    speaker_id=0,
    pitch_scale=1.0,
    speed_scale=1.0,
    energy_scale=1.0
)

# Save to file
tts.save_audio(audio, 'outputs/output.wav')
```

### Voice Cloning

```python
from inference.voice_cloner import VoiceCloner

cloner = VoiceCloner(
    acoustic_checkpoint='checkpoints/acoustic_model.pt',
    vocoder_checkpoint='checkpoints/vocoder_model.pt'
)

# Clone voice from reference audio
audio = cloner.clone_voice(
    text="This is spoken in the cloned voice.",
    reference_audio='path/to/reference.wav',
    language="en"
)

cloner.save_audio(audio, 'outputs/cloned.wav')
```

## Training

### 1. Prepare Dataset

Organize your dataset in the following format:

```
dataset/
├── metadata.csv  # Format: filename|text|speaker_id
└── wavs/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

### 2. Train Acoustic Model

```bash
python training/train_acoustic.py \
    --dataset_path /path/to/dataset \
    --output_dir checkpoints/acoustic \
    --batch_size 32 \
    --epochs 1000 \
    --gpu 0
```

### 3. Train Vocoder

```bash
python training/train_vocoder.py \
    --dataset_path /path/to/dataset \
    --output_dir checkpoints/vocoder \
    --batch_size 16 \
    --epochs 500 \
    --gpu 0
```

## Configuration

All hyperparameters are in `config/`:

- `acoustic_config.py`: Model architecture, attention heads, hidden dimensions
- `vocoder_config.py`: Generator/discriminator settings, upsampling rates
- `training_config.py`: Learning rates, batch sizes, optimization settings

## Performance Optimization

### Speed
- Use mixed precision training (`--fp16`)
- Reduce model size in config files
- Use ONNX export for deployment

### Quality
- Increase model hidden dimensions
- Train longer with more data
- Use higher sampling rate (48kHz)

### Memory
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing

## Supported Languages

The phonemizer supports all languages through Unicode normalization:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Russian (ru)
- Arabic (ar)
- Hindi (hi)
- And many more...

## License

MIT License - Free for commercial and personal use.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{neural_tts_2025,
  title={Neural TTS System - Production Grade},
  author={Soham Vyas},
  year={2025},
  url={https://github.com/vyassoham/neural-tts}
}
```
