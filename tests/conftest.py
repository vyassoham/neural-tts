import pytest
import numpy as np
import torch
import sys
import os

# Ensure the root of the project is in the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Register neural_tts in sys.modules so imports of neural_tts succeed in tests
import __init__ as neural_tts
sys.modules['neural_tts'] = neural_tts

from data.preprocessing import AudioPreprocessor
from inference.synthesizer import TTSSynthesizer
from inference.voice_cloner import VoiceCloner


@pytest.fixture(autouse=True)
def mock_huggingface_hub(mocker):
    """Mock huggingface_hub to prevent network calls during testing"""
    mock_download = mocker.patch("huggingface_hub.hf_hub_download")
    mock_download.return_value = "/mock/cache/path/model.pt"
    return mock_download


@pytest.fixture(autouse=True)
def mock_model_checkpoints(mocker):
    """Mock the checkpoint loading methods to avoid loading from disk"""
    mocker.patch.object(TTSSynthesizer, "_load_acoustic_checkpoint", return_value=None)
    mocker.patch.object(TTSSynthesizer, "_load_vocoder_checkpoint", return_value=None)
    mocker.patch.object(VoiceCloner, "_load_checkpoint", return_value=None)


@pytest.fixture
def mock_audio_preprocessor(mocker):
    """Mock AudioPreprocessor load_audio and save_audio to prevent disk access"""
    mocker.patch.object(
        AudioPreprocessor, 
        "load_audio", 
        return_value=(np.zeros(16000 * 2, dtype=np.float32), 16000)
    )
    mocker.patch.object(AudioPreprocessor, "save_audio", return_value=None)
