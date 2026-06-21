import pytest
import numpy as np
import torch
from inference.synthesizer import TTSSynthesizer
from inference.voice_cloner import VoiceCloner


@pytest.fixture
def mock_duration_predictor(mocker):
    """Mock the duration predictor to always return a positive value for inference stability"""
    # We patch the VariancePredictor forward globally or for model instances.
    # A cleaner way is to mock FastSpeech2's inference method to return a dummy mel spectrogram,
    # or just mock the VariancePredictor.forward method.
    from models.acoustic.variance_adaptor import VariancePredictor
    original_forward = VariancePredictor.forward
    
    def mock_forward(self, x, mask=None):
        # Return a tensor of shape (batch, seq_len) filled with positive duration log values
        batch_size, seq_len, _ = x.shape
        return torch.ones((batch_size, seq_len), device=x.device) * 1.5
        
    mocker.patch.object(VariancePredictor, "forward", mock_forward)


def test_tts_synthesizer_init():
    # Verify instantiation works with mocked checkpoints
    synthesizer = TTSSynthesizer()
    assert synthesizer is not None
    assert synthesizer.device is not None


def test_tts_synthesizer_synthesize(mock_duration_predictor):
    synthesizer = TTSSynthesizer()
    audio = synthesizer.synthesize("Hello world", language='en')
    
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0


def test_tts_synthesizer_batch(mock_duration_predictor):
    synthesizer = TTSSynthesizer()
    texts = ["Hello", "World of speech"]
    audios = synthesizer.synthesize_batch(texts, language='en')
    
    assert isinstance(audios, list)
    assert len(audios) == 2
    assert all(isinstance(a, np.ndarray) for a in audios)


def test_tts_synthesizer_save(mock_duration_predictor, mock_audio_preprocessor):
    synthesizer = TTSSynthesizer()
    audio = np.zeros(16000, dtype=np.float32)
    
    # Verify save_audio runs and prints without filesystem error
    synthesizer.save_audio(audio, "dummy_path.wav")


def test_tts_convenience_speak(mock_duration_predictor, mock_audio_preprocessor):
    from neural_tts import speak
    # Verify speak runs successfully
    res = speak("Hello, this is a convenience function test.", output_file="dummy_out.wav")
    assert res is None


def test_voice_cloner_init():
    cloner = VoiceCloner()
    assert cloner is not None


def test_voice_cloner_extract_embedding(mock_audio_preprocessor):
    cloner = VoiceCloner()
    embedding = cloner.extract_speaker_embedding("dummy_ref.wav")
    
    # Speaker embedding dimension is 256 (from AcousticConfig)
    assert embedding.shape == (256,)


def test_voice_cloner_clone_voice(mock_duration_predictor, mock_audio_preprocessor):
    cloner = VoiceCloner()
    audio = cloner.clone_voice("Hello world", "dummy_ref.wav")
    
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0


def test_voice_cloner_clone_from_embedding(mock_duration_predictor):
    cloner = VoiceCloner()
    embedding = torch.randn(256)
    audio = cloner.clone_voice_from_embedding("Hello world", embedding)
    
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0


def test_voice_cloner_compare_speakers(mock_audio_preprocessor):
    cloner = VoiceCloner()
    similarity = cloner.compare_speakers("dummy_ref1.wav", "dummy_ref2.wav")
    
    assert isinstance(similarity, float)
    assert -1.01 <= similarity <= 1.01
