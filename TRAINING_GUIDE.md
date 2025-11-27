# Complete Training Guide

This guide walks you through training the Neural TTS system from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Training the Acoustic Model](#training-the-acoustic-model)
4. [Training the Vocoder](#training-the-vocoder)
5. [Training the Speaker Encoder (Optional)](#training-the-speaker-encoder-optional)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

---

## Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 16GB
- Storage: 50GB free space

**Recommended:**
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- CPU: 8+ cores
- RAM: 32GB
- Storage: 100GB+ SSD

### Software Requirements

1. **Install Python 3.8+**
   ```bash
   python --version  # Should be 3.8 or higher
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch with CUDA** (for GPU training)
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CPU only
   pip install torch torchvision torchaudio
   ```

4. **Verify Installation**
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

---

## Dataset Preparation

### Dataset Format

Your dataset should be organized as follows:

```
dataset/
├── metadata.csv
└── wavs/
    ├── audio_001.wav
    ├── audio_002.wav
    └── ...
```

### Metadata Format

The `metadata.csv` file should have the following format (no header):

```
filename|text|speaker_id
audio_001.wav|This is the first sentence.|0
audio_002.wav|This is the second sentence.|0
audio_003.wav|Another sentence from speaker two.|1
```

- **filename**: Audio file name (in the `wavs/` directory)
- **text**: Transcription of the audio
- **speaker_id**: Speaker identifier (0 for single-speaker, 0,1,2... for multi-speaker)

### Audio Requirements

- **Format**: WAV, 16-bit PCM
- **Sampling Rate**: 22050 Hz (configurable in `config/vocoder_config.py`)
- **Channels**: Mono
- **Duration**: 1-10 seconds per clip
- **Quality**: Clean audio with minimal background noise

### Preprocessing Audio

Use the following script to preprocess your audio files:

```python
import librosa
import soundfile as sf
import os

def preprocess_audio(input_path, output_path, target_sr=22050):
    """Preprocess audio file"""
    # Load audio
    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
    
    # Normalize
    audio = librosa.util.normalize(audio)
    
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Save
    sf.write(output_path, audio, target_sr)

# Process all files
input_dir = 'raw_audio/'
output_dir = 'dataset/wavs/'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        preprocess_audio(input_path, output_path)
        print(f"Processed: {filename}")
```

### Recommended Datasets

**For English:**
- **LJSpeech**: Single female speaker, 24 hours (~13,100 clips)
  - Download: https://keithito.com/LJ-Speech-Dataset/
- **LibriTTS**: Multi-speaker, 585 hours
  - Download: https://www.openslr.org/60/

**For Multilingual:**
- **Common Voice**: Mozilla's multilingual dataset
  - Download: https://commonvoice.mozilla.org/
- **M-AILABS**: Multi-speaker, multilingual
  - Download: https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/

### Dataset Size Recommendations

- **Minimum**: 1 hour (for testing)
- **Good**: 5-10 hours (decent quality)
- **Excellent**: 20+ hours (high quality)
- **Production**: 50+ hours (professional quality)

---

## Training the Acoustic Model

### Step 1: Configure Training

Edit `config/acoustic_config.py` and `config/training_config.py`:

```python
# config/acoustic_config.py
class AcousticConfig:
    # Adjust based on your dataset
    n_speakers = 1  # Set to number of speakers in your dataset
    
    # For faster training (lower quality)
    encoder_hidden = 128
    encoder_layers = 3
    
    # For better quality (slower training)
    encoder_hidden = 256
    encoder_layers = 4

# config/training_config.py
class TrainingConfig:
    batch_size = 32  # Reduce if out of memory
    epochs = 1000
    acoustic_learning_rate = 1e-4
```

### Step 2: Start Training

```bash
python training/train_acoustic.py \
    --dataset_path dataset/metadata.csv \
    --audio_dir dataset/wavs \
    --output_dir checkpoints/acoustic \
    --batch_size 32 \
    --epochs 1000 \
    --language en \
    --gpu 0
```

**Arguments:**
- `--dataset_path`: Path to metadata CSV
- `--audio_dir`: Directory containing audio files
- `--output_dir`: Where to save checkpoints
- `--batch_size`: Batch size (reduce if OOM)
- `--epochs`: Number of training epochs
- `--language`: Language code (en, es, fr, etc.)
- `--gpu`: GPU device ID (-1 for CPU)

### Step 3: Monitor Training

Training logs are saved to TensorBoard:

```bash
tensorboard --logdir checkpoints/acoustic/logs
```

Open http://localhost:6006 in your browser to view:
- Loss curves
- Learning rate schedule
- Training progress

### Step 4: Training Time Estimates

**On GPU (RTX 3090):**
- 1 hour dataset: ~2-3 hours
- 10 hour dataset: ~12-24 hours
- 50 hour dataset: ~3-5 days

**On CPU:**
- Multiply GPU times by 10-20x

### Step 5: Checkpoints

Checkpoints are saved every 10 epochs:
- `acoustic_model_epoch_10.pt`
- `acoustic_model_epoch_20.pt`
- ...
- `acoustic_model_final.pt`

---

## Training the Vocoder

### Step 1: Configure Training

Edit `config/vocoder_config.py`:

```python
class VocoderConfig:
    # Audio settings
    sampling_rate = 22050
    n_mel_channels = 80
    
    # For faster training
    upsample_initial_channel = 256
    
    # For better quality
    upsample_initial_channel = 512
```

### Step 2: Start Training

```bash
python training/train_vocoder.py \
    --dataset_path dataset/metadata.csv \
    --audio_dir dataset/wavs \
    --output_dir checkpoints/vocoder \
    --batch_size 16 \
    --epochs 500 \
    --gpu 0
```

**Note:** Vocoder training is more memory-intensive. Reduce batch size if needed.

### Step 3: Monitor Training

```bash
tensorboard --logdir checkpoints/vocoder/logs
```

Watch for:
- Generator loss decreasing
- Discriminator loss stabilizing
- Audio quality improving

### Step 4: Training Time Estimates

**On GPU (RTX 3090):**
- 1 hour dataset: ~4-6 hours
- 10 hour dataset: ~24-48 hours
- 50 hour dataset: ~5-7 days

### Step 5: Checkpoints

Checkpoints are saved every 10 epochs:
- `generator_epoch_10.pt`
- `discriminator_epoch_10.pt`
- ...
- `generator_final.pt`

---

## Training the Speaker Encoder (Optional)

For voice cloning, you need to train a speaker encoder.

### Step 1: Prepare Multi-Speaker Dataset

You need a dataset with multiple speakers (at least 10-20 different speakers).

### Step 2: Train Speaker Encoder

```python
# Create training script for speaker encoder
# (Not included in basic system - requires additional implementation)
```

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size:
   ```bash
   --batch_size 16  # or 8, or 4
   ```

2. Reduce model size in config:
   ```python
   encoder_hidden = 128  # instead of 256
   encoder_layers = 3    # instead of 4
   ```

3. Use gradient accumulation (modify training script)

4. Use mixed precision training:
   ```python
   # In training_config.py
   use_fp16 = True
   ```

### Poor Audio Quality

**Solutions:**
1. Train longer (more epochs)
2. Use more/better training data
3. Increase model size
4. Check audio preprocessing
5. Verify dataset quality

### Training is Too Slow

**Solutions:**
1. Use GPU instead of CPU
2. Reduce dataset size for testing
3. Use smaller model
4. Enable mixed precision training
5. Use multiple GPUs (requires code modification)

### Loss Not Decreasing

**Solutions:**
1. Check learning rate (try 1e-3 or 1e-5)
2. Verify dataset format
3. Check for NaN values in data
4. Increase warmup steps
5. Try different optimizer settings

---

## Performance Optimization

### Speed Optimization

1. **Use Mixed Precision:**
   ```python
   use_fp16 = True
   ```

2. **Reduce Model Size:**
   ```python
   encoder_hidden = 128
   decoder_hidden = 128
   ```

3. **Use Smaller Batch Size with Gradient Accumulation**

4. **Export to ONNX for Inference:**
   ```python
   # Export acoustic model
   torch.onnx.export(model, dummy_input, "acoustic.onnx")
   ```

### Quality Optimization

1. **Increase Model Size:**
   ```python
   encoder_hidden = 512
   encoder_layers = 6
   ```

2. **Train Longer:**
   ```python
   epochs = 2000
   ```

3. **Use Higher Sampling Rate:**
   ```python
   sampling_rate = 44100
   ```

4. **Use More Training Data**

5. **Fine-tune on Target Domain**

### Memory Optimization

1. **Gradient Checkpointing:**
   ```python
   # Add to model
   torch.utils.checkpoint.checkpoint(layer, x)
   ```

2. **Reduce Sequence Length:**
   ```python
   max_seq_len = 500  # instead of 1000
   ```

3. **Use CPU Offloading for Large Models**

---

## Next Steps

After training:

1. **Test Inference:**
   ```bash
   python examples/example_synthesis.py
   ```

2. **Evaluate Quality:**
   - Listen to generated samples
   - Compare with ground truth
   - Measure MOS (Mean Opinion Score)

3. **Fine-tune:**
   - Adjust hyperparameters
   - Add more data
   - Try different configurations

4. **Deploy:**
   - Export to ONNX
   - Create API server
   - Optimize for production

---

## Additional Resources

- **Paper: FastSpeech 2**: https://arxiv.org/abs/2006.04558
- **Paper: HiFi-GAN**: https://arxiv.org/abs/2010.05646
- **PyTorch Documentation**: https://pytorch.org/docs/
- **TensorBoard Guide**: https://www.tensorflow.org/tensorboard

---

**Good luck with your training! 🚀**
