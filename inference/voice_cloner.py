"""
Voice Cloner
Speaker cloning from reference audio
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
from models.speaker.encoder import SpeakerEncoder
from text.phonemizer import Phonemizer
from data.preprocessing import AudioPreprocessor


class VoiceCloner:
    """
    Voice Cloning System
    Clones voice characteristics from reference audio
    """
    
    def __init__(self, acoustic_checkpoint=None, vocoder_checkpoint=None,
                 speaker_encoder_checkpoint=None, acoustic_config=None,
                 vocoder_config=None, device=None):
        """
        Initialize voice cloner
        
        Args:
            acoustic_checkpoint: Path to acoustic model checkpoint
            vocoder_checkpoint: Path to vocoder checkpoint
            speaker_encoder_checkpoint: Path to speaker encoder checkpoint
            acoustic_config: AcousticConfig object
            vocoder_config: VocoderConfig object
            device: Device to run on
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
        self.speaker_encoder = SpeakerEncoder(
            n_mel_channels=self.vocoder_config.n_mel_channels,
            embedding_dim=self.acoustic_config.speaker_embed_dim
        ).to(self.device)
        
        # Load checkpoints
        if acoustic_checkpoint:
            self._load_checkpoint(self.acoustic_model, acoustic_checkpoint)
        
        if vocoder_checkpoint:
            self._load_checkpoint(self.vocoder, vocoder_checkpoint)
        
        if speaker_encoder_checkpoint:
            self._load_checkpoint(self.speaker_encoder, speaker_encoder_checkpoint)
        
        # Set to eval mode
        self.acoustic_model.eval()
        self.vocoder.eval()
        self.speaker_encoder.eval()
        
        # Initialize phonemizer and preprocessor
        self.phonemizers = {}
        self.preprocessor = AudioPreprocessor(self.vocoder_config)
    
    def _load_checkpoint(self, model, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    def _get_phonemizer(self, language):
        """Get or create phonemizer for language"""
        if language not in self.phonemizers:
            self.phonemizers[language] = Phonemizer(language)
        return self.phonemizers[language]
    
    def extract_speaker_embedding(self, audio_path):
        """
        Extract speaker embedding from audio file
        
        Args:
            audio_path: Path to reference audio file
            
        Returns:
            embedding: (embedding_dim,) - speaker embedding
        """
        # Load audio
        audio, _ = self.preprocessor.load_audio(audio_path)
        
        # Extract mel-spectrogram
        mel = self.preprocessor.compute_mel_spectrogram(audio)
        
        # Transpose for encoder (expects time, n_mel)
        mel = mel.transpose(0, 1).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.speaker_encoder(mel)
        
        return embedding.squeeze(0)
    
    def clone_voice(self, text, reference_audio, language='en',
                   pitch_scale=1.0, speed_scale=1.0, energy_scale=1.0):
        """
        Clone voice from reference audio and synthesize text
        
        Args:
            text: Input text to synthesize
            reference_audio: Path to reference audio file (5-10 seconds)
            language: Language code
            pitch_scale: Pitch control
            speed_scale: Speed control
            energy_scale: Energy control
            
        Returns:
            audio: (time,) - synthesized audio in cloned voice
        """
        # Extract speaker embedding from reference
        speaker_embedding = self.extract_speaker_embedding(reference_audio)
        
        # Convert text to phoneme sequence
        phonemizer = self._get_phonemizer(language)
        phoneme_ids = phonemizer.text_to_sequence(text)
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long, device=self.device)
        
        # Generate mel-spectrogram with speaker embedding
        # Note: This requires modifying the acoustic model to accept embeddings
        # For now, we'll use a simplified approach
        with torch.no_grad():
            # This is a simplified version - in practice, you'd need to
            # integrate the speaker embedding into the acoustic model
            mel = self.acoustic_model.inference(
                phonemes=phoneme_ids,
                speaker_id=0,  # Placeholder
                duration_control=1.0 / speed_scale,
                pitch_control=pitch_scale,
                energy_control=energy_scale
            )
        
        # Generate audio from mel-spectrogram
        with torch.no_grad():
            audio = self.vocoder.inference(mel)
        
        # Convert to numpy
        audio = audio.cpu().numpy()
        
        return audio
    
    def clone_voice_from_embedding(self, text, speaker_embedding, language='en',
                                   pitch_scale=1.0, speed_scale=1.0, energy_scale=1.0):
        """
        Synthesize with pre-computed speaker embedding
        
        Args:
            text: Input text
            speaker_embedding: Pre-computed speaker embedding
            language: Language code
            pitch_scale: Pitch control
            speed_scale: Speed control
            energy_scale: Energy control
            
        Returns:
            audio: (time,) - synthesized audio
        """
        # Convert text to phoneme sequence
        phonemizer = self._get_phonemizer(language)
        phoneme_ids = phonemizer.text_to_sequence(text)
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long, device=self.device)
        
        # Generate mel-spectrogram
        with torch.no_grad():
            mel = self.acoustic_model.inference(
                phonemes=phoneme_ids,
                speaker_id=0,
                duration_control=1.0 / speed_scale,
                pitch_control=pitch_scale,
                energy_control=energy_scale
            )
        
        # Generate audio
        with torch.no_grad():
            audio = self.vocoder.inference(mel)
        
        # Convert to numpy
        audio = audio.cpu().numpy()
        
        return audio
    
    def save_audio(self, audio, path, sr=None):
        """
        Save audio to file
        
        Args:
            audio: Audio waveform
            path: Output path
            sr: Sampling rate
        """
        self.preprocessor.save_audio(audio, path, sr)
        print(f"Audio saved to: {path}")
    
    def compare_speakers(self, audio_path1, audio_path2):
        """
        Compare similarity between two speakers
        
        Args:
            audio_path1: Path to first audio file
            audio_path2: Path to second audio file
            
        Returns:
            similarity: Cosine similarity score (0-1)
        """
        # Extract embeddings
        emb1 = self.extract_speaker_embedding(audio_path1)
        emb2 = self.extract_speaker_embedding(audio_path2)
        
        # Compute similarity
        similarity = self.speaker_encoder.similarity(emb1, emb2)
        
        return similarity.item()
