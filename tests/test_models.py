import pytest
import torch
import numpy as np
from config.acoustic_config import AcousticConfig
from config.vocoder_config import VocoderConfig
from models.acoustic.fastspeech2 import FastSpeech2
from models.vocoder.hifigan import HiFiGANGenerator
from models.speaker.encoder import SpeakerEncoder, SpeakerEmbeddingTable


def test_fastspeech2_forward():
    config = AcousticConfig()
    config.n_speakers = 2
    model = FastSpeech2(config)
    
    # Input phoneme IDs: batch=2, seq_len=10
    phonemes = torch.randint(1, config.vocab_size, (2, 10))
    speaker_ids = torch.randint(0, config.n_speakers, (2,))
    
    # Explicit durations to ensure deterministic output length
    durations = torch.ones((2, 10), dtype=torch.long) * 3
    pitch = torch.randn(2, 10)
    energy = torch.randn(2, 10)
    
    mel_output, duration_pred, pitch_pred, energy_pred = model(
        phonemes, 
        speaker_ids=speaker_ids,
        durations=durations,
        pitch=pitch,
        energy=energy
    )
    
    # Check output shapes
    # Max length should be 10 * 3 = 30
    assert mel_output.shape == (2, 30, config.n_mel_channels)
    assert duration_pred.shape == (2, 10)
    assert pitch_pred.shape == (2, 10)
    assert energy_pred.shape == (2, 10)


def test_fastspeech2_inference(mocker):
    config = AcousticConfig()
    model = FastSpeech2(config)
    
    # Mock duration_predictor forward to return high values so durations are positive
    mocker.patch.object(
        model.variance_adaptor.duration_predictor,
        "forward",
        return_value=torch.ones((1, 5)) * 1.5
    )
    
    phonemes = torch.randint(1, config.vocab_size, (5,))
    
    mel_output = model.inference(phonemes)
    
    assert mel_output.dim() == 2
    assert mel_output.size(1) == config.n_mel_channels
    assert mel_output.size(0) > 0  # Should be non-zero length


def test_hifigan_generator():
    config = VocoderConfig()
    generator = HiFiGANGenerator(config)
    
    # Mel-spectrogram: batch=2, channels=80, time=16
    mel = torch.randn(2, config.n_mel_channels, 16)
    
    audio = generator(mel)
    
    # Output audio shape: (batch, 1, time * hop_length)
    # hop_length = 256
    expected_length = 16 * config.hop_length
    assert audio.shape == (2, 1, expected_length)


def test_hifigan_inference():
    config = VocoderConfig()
    generator = HiFiGANGenerator(config)
    
    mel = torch.randn(16, config.n_mel_channels)
    
    audio = generator.inference(mel)
    
    # For single sequence input, output should be 1D waveform
    expected_length = 16 * config.hop_length
    assert audio.shape == (expected_length,)


def test_speaker_encoder():
    encoder = SpeakerEncoder(n_mel_channels=80, embedding_dim=256, lstm_hidden=64, num_layers=2)
    
    # Mel-spectrogram input: batch=3, time=40, channels=80
    mels = torch.randn(3, 40, 80)
    
    embeddings = encoder(mels)
    
    assert embeddings.shape == (3, 256)
    # Check L2 normalized
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_speaker_encoder_compute_embedding():
    encoder = SpeakerEncoder(n_mel_channels=80, embedding_dim=256, lstm_hidden=64, num_layers=2)
    mel = torch.randn(30, 80)
    
    embedding = encoder.compute_embedding(mel)
    
    assert embedding.shape == (256,)


def test_speaker_encoder_similarity():
    encoder = SpeakerEncoder(n_mel_channels=80, embedding_dim=256, lstm_hidden=64, num_layers=2)
    
    emb1 = torch.randn(256)
    emb2 = torch.randn(256)
    
    sim = encoder.similarity(emb1, emb2)
    assert sim.dim() == 0  # Scalar similarity
    
    # Similarity of identical embeddings should be 1.0
    sim_self = encoder.similarity(emb1, emb1)
    assert torch.allclose(sim_self, torch.tensor(1.0), atol=1e-5)


def test_speaker_embedding_table():
    table = SpeakerEmbeddingTable(num_speakers=5, embedding_dim=256)
    speaker_ids = torch.tensor([0, 2, 4])
    
    embeddings = table(speaker_ids)
    
    assert embeddings.shape == (3, 256)
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
