"""
TTS Synthesizer
Main inference engine for text-to-speech synthesis
"""

import torch
import numpy as np
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.acoustic_config import AcousticConfig
from config.vocoder_config import VocoderConfig
from models.acoustic.fastspeech2 import FastSpeech2
from models.vocoder.hifigan import HiFiGANGenerator
from text.phonemizer import Phonemizer
from data.preprocessing import AudioPreprocessor


class TTSSynthesizer:
    """
    Text-to-Speech Synthesizer
    Combines acoustic model and vocoder for end-to-end synthesis
    """
    
    def __init__(self, acoustic_checkpoint=None, vocoder_checkpoint=None,
                 acoustic_config=None, vocoder_config=None, device=None):
        """
        Initialize TTS synthesizer
        
        Args:
            acoustic_checkpoint: Path to acoustic model checkpoint
            vocoder_checkpoint: Path to vocoder checkpoint
            acoustic_config: AcousticConfig object (None = use default)
            vocoder_config: VocoderConfig object (None = use default)
            device: Device to run on (None = auto-detect)
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load configs
        self.acoustic_config = acoustic_config if acoustic_config else AcousticConfig()
        self.vocoder_config = vocoder_config if vocoder_config else VocoderConfig()
        
        # Initialize models
        self.acoustic_model = FastSpeech2(self.acoustic_config).to(self.device)
        self.vocoder = HiFiGANGenerator(self.vocoder_config).to(self.device)
        
        # Load checkpoints
        if acoustic_checkpoint:
            self._load_acoustic_checkpoint(acoustic_checkpoint)
        
        if vocoder_checkpoint:
            self._load_vocoder_checkpoint(vocoder_checkpoint)
        
        # Set to eval mode
        self.acoustic_model.eval()
        self.vocoder.eval()
        
        # Initialize phonemizer and preprocessor
        self.phonemizers = {}  # Cache phonemizers for different languages
        self.preprocessor = AudioPreprocessor(self.vocoder_config)
    
    def _load_acoustic_checkpoint(self, checkpoint_path):
        """Load acoustic model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.acoustic_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded acoustic model from: {checkpoint_path}")
    
    def _load_vocoder_checkpoint(self, checkpoint_path):
        """Load vocoder checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.vocoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded vocoder from: {checkpoint_path}")
    
    def _get_phonemizer(self, language):
        """Get or create phonemizer for language"""
        if language not in self.phonemizers:
            self.phonemizers[language] = Phonemizer(language)
        return self.phonemizers[language]
    
    def synthesize(self, text, language='en', speaker_id=0,
                  pitch_scale=1.0, speed_scale=1.0, energy_scale=1.0):
        """
        Synthesize speech from text
        
        Args:
            text: Input text
            language: Language code
            speaker_id: Speaker ID (for multi-speaker models)
            pitch_scale: Pitch control (1.0 = normal)
            speed_scale: Speed control (1.0 = normal, <1 = slower, >1 = faster)
            energy_scale: Energy/volume control (1.0 = normal)
            
        Returns:
            audio: (time,) - synthesized audio waveform
        """
        # Convert text to phoneme sequence
        phonemizer = self._get_phonemizer(language)
        phoneme_ids = phonemizer.text_to_sequence(text)
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long, device=self.device)
        
        # Generate mel-spectrogram
        with torch.no_grad():
            mel = self.acoustic_model.inference(
                phonemes=phoneme_ids,
                speaker_id=speaker_id,
                duration_control=1.0 / speed_scale,  # Inverse for speed
                pitch_control=pitch_scale,
                energy_control=energy_scale
            )
        
        # Generate audio from mel-spectrogram
        with torch.no_grad():
            audio = self.vocoder.inference(mel)
        
        # Convert to numpy
        audio = audio.cpu().numpy()
        
        return audio
    
    def synthesize_batch(self, texts, language='en', speaker_ids=None,
                        pitch_scales=None, speed_scales=None, energy_scales=None):
        """
        Synthesize speech from multiple texts
        
        Args:
            texts: List of input texts
            language: Language code
            speaker_ids: List of speaker IDs (None = all 0)
            pitch_scales: List of pitch scales (None = all 1.0)
            speed_scales: List of speed scales (None = all 1.0)
            energy_scales: List of energy scales (None = all 1.0)
            
        Returns:
            audios: List of synthesized audio waveforms
        """
        # Set defaults
        batch_size = len(texts)
        if speaker_ids is None:
            speaker_ids = [0] * batch_size
        if pitch_scales is None:
            pitch_scales = [1.0] * batch_size
        if speed_scales is None:
            speed_scales = [1.0] * batch_size
        if energy_scales is None:
            energy_scales = [1.0] * batch_size
        
        # Synthesize each text
        audios = []
        for text, spk_id, pitch, speed, energy in zip(
            texts, speaker_ids, pitch_scales, speed_scales, energy_scales
        ):
            audio = self.synthesize(
                text, language, spk_id, pitch, speed, energy
            )
            audios.append(audio)
        
        return audios
    
    def save_audio(self, audio, path, sr=None):
        """
        Save audio to file
        
        Args:
            audio: Audio waveform
            path: Output path
            sr: Sampling rate (None = use config)
        """
        self.preprocessor.save_audio(audio, path, sr)
        print(f"Audio saved to: {path}")
    
    def text_to_mel(self, text, language='en', speaker_id=0,
                   pitch_scale=1.0, speed_scale=1.0, energy_scale=1.0):
        """
        Convert text to mel-spectrogram only
        
        Args:
            text: Input text
            language: Language code
            speaker_id: Speaker ID
            pitch_scale: Pitch control
            speed_scale: Speed control
            energy_scale: Energy control
            
        Returns:
            mel: (n_mel, time) - mel-spectrogram
        """
        # Convert text to phoneme sequence
        phonemizer = self._get_phonemizer(language)
        phoneme_ids = phonemizer.text_to_sequence(text)
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long, device=self.device)
        
        # Generate mel-spectrogram
        with torch.no_grad():
            mel = self.acoustic_model.inference(
                phonemes=phoneme_ids,
                speaker_id=speaker_id,
                duration_control=1.0 / speed_scale,
                pitch_control=pitch_scale,
                energy_control=energy_scale
            )
        
        return mel
    
    def mel_to_audio(self, mel):
        """
        Convert mel-spectrogram to audio
        
        Args:
            mel: (n_mel, time) or (batch, n_mel, time) - mel-spectrogram
            
        Returns:
            audio: (time,) or (batch, time) - audio waveform
        """
        # Ensure tensor
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel).float().to(self.device)
        
        # Generate audio
        with torch.no_grad():
            audio = self.vocoder.inference(mel)
        
        # Convert to numpy
        if audio.dim() == 1:
            audio = audio.cpu().numpy()
        else:
            audio = [a.cpu().numpy() for a in audio]
        
        return audio
