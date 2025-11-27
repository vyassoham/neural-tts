# Neural TTS System Architecture

Complete technical documentation of the Neural TTS system architecture.

## Table of Contents

1. [System Overview](#system-overview)
2. [Text Processing Pipeline](#text-processing-pipeline)
3. [Acoustic Model (FastSpeech2)](#acoustic-model-fastspeech2)
4. [Vocoder (HiFi-GAN)](#vocoder-hifi-gan)
5. [Speaker Encoder](#speaker-encoder)
6. [Training Pipeline](#training-pipeline)
7. [Inference Pipeline](#inference-pipeline)
8. [Performance Characteristics](#performance-characteristics)

---

## System Overview

### Architecture Diagram

```
Input Text
    ↓
[Text Normalizer] → Unicode normalization, abbreviation expansion
    ↓
[Phonemizer] → Text to phoneme conversion
    ↓
[Phoneme Encoder] → Phoneme IDs
    ↓
[FastSpeech2 Acoustic Model]
    ├─ Phoneme Embedding
    ├─ Encoder (Transformer)
    ├─ Variance Adaptor
    │   ├─ Duration Predictor
    │   ├─ Pitch Predictor
    │   └─ Energy Predictor
    ├─ Length Regulator
    └─ Decoder (Transformer)
    ↓
Mel-Spectrogram
    ↓
[HiFi-GAN Vocoder]
    ├─ Generator
    │   ├─ Upsampling Layers
    │   └─ Multi-Receptive Field Fusion
    └─ Discriminators (training only)
        ├─ Multi-Period Discriminator
        └─ Multi-Scale Discriminator
    ↓
Audio Waveform
```

### Key Components

1. **Text Processing**: Converts raw text to phoneme sequences
2. **Acoustic Model**: Generates mel-spectrograms from phonemes
3. **Vocoder**: Converts mel-spectrograms to audio waveforms
4. **Speaker Encoder**: Extracts speaker embeddings for voice cloning

---

## Text Processing Pipeline

### 1. Text Normalizer

**Purpose**: Clean and standardize input text

**Operations:**
- Unicode normalization (NFKC)
- Lowercase conversion
- Abbreviation expansion
- Number to word conversion
- Special character handling

**Example:**
```
Input:  "Dr. Smith said it's 5:30 PM."
Output: "doctor smith said it's five thirty pee em"
```

**Implementation**: `text/normalizer.py`

### 2. Phonemizer

**Purpose**: Convert text to phoneme sequences

**Methods:**
1. **Rule-based**: Simple grapheme-to-phoneme rules
2. **Dictionary-based**: Lookup common words
3. **External**: Use espeak/phonemizer library (optional)

**Phoneme Set:**
- ARPAbet for English (AA, AE, AH, etc.)
- IPA for multilingual support
- Special tokens (padding, BOS, EOS)

**Example:**
```
Input:  "hello world"
Output: ['HH', 'AH', 'L', 'OW', ' ', 'W', 'ER', 'L', 'D']
```

**Implementation**: `text/phonemizer.py`

### 3. Symbol Mapping

**Purpose**: Convert phonemes to model-ready IDs

**Process:**
```
Phonemes → Symbol IDs → Tensor
['HH', 'AH', 'L', 'OW'] → [23, 15, 31, 42] → torch.tensor([23, 15, 31, 42])
```

**Implementation**: `text/symbols.py`

---

## Acoustic Model (FastSpeech2)

### Architecture Overview

FastSpeech2 is a non-autoregressive transformer-based model that generates mel-spectrograms from phoneme sequences.

### Components

#### 1. Phoneme Embedding

**Purpose**: Convert phoneme IDs to dense vectors

```python
Input:  [23, 15, 31, 42]  # Phoneme IDs
        ↓
Embedding Layer (vocab_size=512, hidden_dim=256)
        ↓
Output: (batch, seq_len, 256)  # Dense vectors
```

#### 2. Encoder

**Purpose**: Extract linguistic features from phonemes

**Architecture:**
- 4 Feed-Forward Transformer (FFT) blocks
- Each block contains:
  - Multi-head self-attention (2 heads)
  - Position-wise feed-forward network
  - Layer normalization
  - Residual connections

**Attention Mechanism:**
```
Query, Key, Value ← Linear projections of input
Attention scores = softmax(QK^T / √d_k)
Output = Attention scores × Value
```

**Implementation**: `models/acoustic/transformer.py`

#### 3. Variance Adaptor

**Purpose**: Predict and control prosody (duration, pitch, energy)

**Sub-components:**

**a) Duration Predictor**
- Predicts phoneme durations (how long each phoneme lasts)
- Architecture: 2 Conv1D layers + Linear
- Output: Log-domain durations
- Loss: MSE between predicted and ground truth

**b) Pitch Predictor**
- Predicts fundamental frequency (F0)
- Quantized into bins for embedding
- Controls voice pitch
- Loss: MSE between predicted and ground truth

**c) Energy Predictor**
- Predicts frame-level energy (volume)
- Quantized into bins for embedding
- Controls loudness
- Loss: MSE between predicted and ground truth

**d) Length Regulator**
- Expands phoneme-level features to frame-level
- Uses predicted durations
- Enables speed control

**Example:**
```
Phoneme: ['H', 'E', 'L', 'L', 'O']
Durations: [3, 4, 2, 2, 5]  # frames per phoneme
         ↓
Frame-level: ['H','H','H', 'E','E','E','E', 'L','L', 'L','L', 'O','O','O','O','O']
```

**Implementation**: `models/acoustic/variance_adaptor.py`

#### 4. Decoder

**Purpose**: Generate mel-spectrogram from frame-level features

**Architecture:**
- 4 FFT blocks (same as encoder)
- Final linear layer projects to mel-spectrogram
- Output: (batch, time, n_mel_channels)

**Implementation**: `models/acoustic/fastspeech2.py`

### Training

**Inputs:**
- Phoneme sequences
- Ground truth mel-spectrograms
- Ground truth durations (from forced alignment)
- Ground truth pitch
- Ground truth energy

**Losses:**
```
Total Loss = λ_mel × L_mel + λ_dur × L_dur + λ_pitch × L_pitch + λ_energy × L_energy

Where:
- L_mel: L1 loss between predicted and ground truth mel
- L_dur: MSE loss for duration prediction
- L_pitch: MSE loss for pitch prediction
- L_energy: MSE loss for energy prediction
```

**Optimizer:** Adam with Noam learning rate schedule

---

## Vocoder (HiFi-GAN)

### Architecture Overview

HiFi-GAN is a generative adversarial network that converts mel-spectrograms to high-fidelity audio waveforms.

### Generator

**Purpose**: Convert mel-spectrogram to waveform

**Architecture:**

1. **Pre-convolution**: Initial 1D convolution
   ```
   Input: (batch, 80, time)  # Mel-spectrogram
   Conv1D(80 → 512, kernel=7)
   Output: (batch, 512, time)
   ```

2. **Upsampling Blocks**: 4 transposed convolutions
   ```
   Upsample rates: [8, 8, 2, 2]  # Product = 256 (hop_length)
   
   Block 1: 512 → 256, upsample 8x
   Block 2: 256 → 128, upsample 8x
   Block 3: 128 → 64, upsample 2x
   Block 4: 64 → 32, upsample 2x
   ```

3. **Multi-Receptive Field Fusion (MRF)**
   - After each upsampling block
   - Multiple residual blocks with different kernel sizes
   - Captures patterns at different scales
   - Kernel sizes: [3, 7, 11]
   - Dilation rates: [[1,3,5], [1,3,5], [1,3,5]]

4. **Post-convolution**: Final 1D convolution
   ```
   Conv1D(32 → 1, kernel=7)
   Tanh activation
   Output: (batch, 1, time × 256)  # Waveform
   ```

**Implementation**: `models/vocoder/hifigan.py`

### Discriminators

**Purpose**: Distinguish real from generated audio (training only)

#### Multi-Period Discriminator (MPD)

**Concept**: Analyze audio at different periods

**Periods**: [2, 3, 5, 7, 11]

**Process:**
```
Audio: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Period 2: [[1,3,5,7,9,11], [2,4,6,8,10,12]]
Period 3: [[1,4,7,10], [2,5,8,11], [3,6,9,12]]
...
```

**Architecture per period:**
- 5 2D convolution layers
- LeakyReLU activations
- Output: Real/fake score

#### Multi-Scale Discriminator (MSD)

**Concept**: Analyze audio at different resolutions

**Scales**: 3 (original, 2x downsampled, 4x downsampled)

**Architecture per scale:**
- 6 1D convolution layers
- Grouped convolutions
- LeakyReLU activations
- Output: Real/fake score

**Implementation**: `models/vocoder/discriminator.py`

### Training

**Adversarial Training:**

1. **Discriminator Step:**
   ```
   L_D = E[((D(real) - 1)^2] + E[(D(fake))^2]
   ```

2. **Generator Step:**
   ```
   L_G = λ_adv × L_adv + λ_feat × L_feat + λ_mel × L_mel
   
   Where:
   - L_adv: Adversarial loss (fool discriminator)
   - L_feat: Feature matching loss (match intermediate features)
   - L_mel: Mel-spectrogram reconstruction loss
   ```

**Optimizer:** Adam with constant learning rate

---

## Speaker Encoder

### Architecture

**Purpose**: Extract speaker identity from audio

**Architecture:**
- 3-layer LSTM
- Input: Mel-spectrogram (time, 80)
- Hidden: 768 dimensions
- Output: 256-dimensional embedding

**Process:**
```
Mel-spectrogram
    ↓
LSTM layers (3 × 768 hidden)
    ↓
Average pooling over time
    ↓
Linear projection (768 → 256)
    ↓
L2 normalization
    ↓
Speaker embedding (256-dim)
```

**Training:**
- Generalized End-to-End (GE2E) loss
- Contrastive learning
- Same speaker: high similarity
- Different speakers: low similarity

**Implementation**: `models/speaker/encoder.py`

---

## Training Pipeline

### Data Flow

```
Raw Audio + Text
    ↓
[Preprocessing]
    ├─ Audio: Normalize, trim silence, resample
    └─ Text: Normalize, phonemize
    ↓
[Feature Extraction]
    ├─ Mel-spectrogram
    ├─ Pitch (F0)
    ├─ Energy
    └─ Duration (from forced alignment)
    ↓
[Dataset]
    ↓
[DataLoader] → Batching, padding, collation
    ↓
[Model Training]
    ├─ Forward pass
    ├─ Loss computation
    ├─ Backward pass
    └─ Optimizer step
    ↓
[Checkpointing]
```

### Training Stages

**Stage 1: Acoustic Model**
- Input: Phonemes + ground truth features
- Output: Mel-spectrograms
- Duration: ~1000 epochs
- Time: 1-5 days (depending on dataset)

**Stage 2: Vocoder**
- Input: Mel-spectrograms
- Output: Audio waveforms
- Duration: ~500 epochs
- Time: 1-7 days (depending on dataset)

**Stage 3: Speaker Encoder (Optional)**
- Input: Audio from multiple speakers
- Output: Speaker embeddings
- Duration: ~200 epochs
- Time: 1-3 days

---

## Inference Pipeline

### End-to-End Flow

```
Input Text: "Hello, world!"
    ↓
[Text Normalization]
    ↓
Normalized: "hello world"
    ↓
[Phonemization]
    ↓
Phonemes: ['HH', 'AH', 'L', 'OW', ' ', 'W', 'ER', 'L', 'D']
    ↓
[Symbol Mapping]
    ↓
IDs: [23, 15, 31, 42, 1, 56, 28, 31, 19]
    ↓
[FastSpeech2]
    ├─ Phoneme Embedding
    ├─ Encoder
    ├─ Variance Adaptor
    │   ├─ Predict durations
    │   ├─ Predict pitch
    │   └─ Predict energy
    ├─ Length Regulation
    └─ Decoder
    ↓
Mel-Spectrogram: (80, 150)
    ↓
[HiFi-GAN]
    ├─ Upsampling
    └─ MRF
    ↓
Audio Waveform: (38400,)  # 150 × 256 samples
    ↓
Output: "hello_world.wav"
```

### Latency Analysis

**On GPU (RTX 3090):**
- Text processing: ~1-5 ms
- Acoustic model: ~10-50 ms (depends on text length)
- Vocoder: ~20-100 ms (depends on mel length)
- **Total: ~30-155 ms** (real-time capable)

**On CPU (Intel i7):**
- Text processing: ~5-10 ms
- Acoustic model: ~100-500 ms
- Vocoder: ~200-1000 ms
- **Total: ~305-1510 ms** (not real-time)

---

## Performance Characteristics

### Model Sizes

**Acoustic Model (FastSpeech2):**
- Parameters: ~10-30M (depends on config)
- Memory: ~100-300 MB
- Inference: ~10-50 ms/sentence (GPU)

**Vocoder (HiFi-GAN):**
- Parameters: ~13M
- Memory: ~50-100 MB
- Inference: ~20-100 ms/second of audio (GPU)

**Speaker Encoder:**
- Parameters: ~5M
- Memory: ~20-50 MB
- Inference: ~5-10 ms/audio (GPU)

### Audio Quality

**Metrics:**
- **MOS (Mean Opinion Score)**: 4.0-4.5 / 5.0
- **Naturalness**: High (comparable to human speech)
- **Intelligibility**: >95%
- **Speaker similarity**: >0.85 (for voice cloning)

### Scalability

**Batch Processing:**
- GPU: 32-64 samples simultaneously
- CPU: 4-8 samples simultaneously

**Throughput:**
- GPU: ~100-500 sentences/minute
- CPU: ~10-50 sentences/minute

---

## Optimization Techniques

### Speed Optimization

1. **Model Quantization**: INT8 quantization (2-4x speedup)
2. **ONNX Export**: Optimized runtime (1.5-2x speedup)
3. **TensorRT**: GPU optimization (2-3x speedup)
4. **Mixed Precision**: FP16 training/inference (1.5-2x speedup)

### Quality Optimization

1. **Larger Models**: More parameters = better quality
2. **More Training Data**: 50+ hours recommended
3. **Better Preprocessing**: Clean audio, accurate transcriptions
4. **Fine-tuning**: Domain-specific adaptation

### Memory Optimization

1. **Gradient Checkpointing**: Reduce memory during training
2. **Smaller Batch Sizes**: Trade speed for memory
3. **Model Pruning**: Remove unnecessary parameters
4. **CPU Offloading**: Move parts to CPU

---

## References

### Papers

1. **FastSpeech 2**: Ren et al., 2020
   - https://arxiv.org/abs/2006.04558

2. **HiFi-GAN**: Kong et al., 2020
   - https://arxiv.org/abs/2010.05646

3. **Transformer**: Vaswani et al., 2017
   - https://arxiv.org/abs/1706.03762

4. **GE2E Speaker Encoder**: Wan et al., 2018
   - https://arxiv.org/abs/1710.10467

### Code Structure

```
tts/
├── config/          # Configuration files
├── text/            # Text processing
├── models/          # Neural network models
│   ├── acoustic/    # FastSpeech2
│   ├── vocoder/     # HiFi-GAN
│   └── speaker/     # Speaker encoder
├── data/            # Dataset and preprocessing
├── training/        # Training scripts
├── inference/       # Inference engines
├── utils/           # Utilities
└── examples/        # Usage examples
```

---

**This architecture enables:**
- ✅ High-quality speech synthesis
- ✅ Real-time inference
- ✅ Multilingual support
- ✅ Voice cloning
- ✅ Prosody control
- ✅ Production-ready deployment
