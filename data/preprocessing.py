"""
Audio Preprocessing Utilities
Handles audio loading, mel-spectrogram extraction, and feature computation
"""

import torch
import torchaudio
import librosa
import numpy as np
from scipy import signal


class AudioPreprocessor:
    """Audio preprocessing for TTS"""
    
    def __init__(self, config):
        """
        Initialize audio preprocessor
        
        Args:
            config: VocoderConfig object
        """
        self.config = config
        self.sampling_rate = config.sampling_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.n_mel_channels = config.n_mel_channels
        self.mel_fmin = config.mel_fmin
        self.mel_fmax = config.mel_fmax if config.mel_fmax else config.sampling_rate / 2
        
        # Create mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()
    
    def load_audio(self, path, target_sr=None):
        """
        Load audio file
        
        Args:
            path: Path to audio file
            target_sr: Target sampling rate (None = use config)
            
        Returns:
            audio: (time,) - audio waveform
            sr: Sampling rate
        """
        if target_sr is None:
            target_sr = self.sampling_rate
        
        # Load audio
        audio, sr = librosa.load(path, sr=target_sr, mono=True)
        
        return audio, sr
    
    def save_audio(self, audio, path, sr=None):
        """
        Save audio file
        
        Args:
            audio: (time,) - audio waveform
            path: Output path
            sr: Sampling rate (None = use config)
        """
        if sr is None:
            sr = self.sampling_rate
        
        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Ensure audio is in valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        # Save
        import soundfile as sf
        sf.write(path, audio, sr)
    
    def normalize_audio(self, audio, target_db=-20):
        """
        Normalize audio to target dB
        
        Args:
            audio: (time,) - audio waveform
            target_db: Target dB level
            
        Returns:
            normalized_audio: (time,)
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Convert to dB
        current_db = 20 * np.log10(rms + 1e-8)
        
        # Calculate gain
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio * gain
        
        # Clip to prevent clipping
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def compute_stft(self, audio):
        """
        Compute STFT
        
        Args:
            audio: (time,) - audio waveform
            
        Returns:
            stft: (n_fft // 2 + 1, time) - complex STFT
        """
        # Convert to tensor if numpy
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length),
            return_complex=True
        )
        
        return stft
    
    def compute_mel_spectrogram(self, audio):
        """
        Compute mel-spectrogram
        
        Args:
            audio: (time,) or (batch, time) - audio waveform
            
        Returns:
            mel: (n_mel, time) or (batch, n_mel, time) - mel-spectrogram
        """
        # Handle batch
        single_input = False
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            single_input = True
        
        batch_size = audio.size(0)
        mels = []
        
        for i in range(batch_size):
            # Compute STFT
            stft = torch.stft(
                audio[i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(self.win_length, device=audio.device),
                return_complex=True
            )
            
            # Magnitude
            magnitude = torch.abs(stft)
            
            # Apply mel filterbank
            mel = torch.matmul(self.mel_basis.to(audio.device), magnitude)
            
            # Log scale
            mel = torch.log(torch.clamp(mel, min=1e-5))
            
            mels.append(mel)
        
        mel = torch.stack(mels, dim=0)
        
        if single_input:
            mel = mel.squeeze(0)
        
        return mel
    
    def extract_pitch(self, audio, sr=None):
        """
        Extract pitch (F0) from audio
        
        Args:
            audio: (time,) - audio waveform
            sr: Sampling rate
            
        Returns:
            pitch: (time // hop_length,) - pitch contour in Hz
        """
        if sr is None:
            sr = self.sampling_rate
        
        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Extract pitch using librosa
        pitch, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=self.hop_length
        )
        
        # Interpolate unvoiced regions
        pitch = self._interpolate_pitch(pitch)
        
        # Normalize to [-1, 1] range (log scale)
        pitch_log = np.log(pitch + 1e-8)
        pitch_normalized = (pitch_log - pitch_log.mean()) / (pitch_log.std() + 1e-8)
        
        return torch.from_numpy(pitch_normalized).float()
    
    def _interpolate_pitch(self, pitch):
        """Interpolate unvoiced pitch values"""
        # Find voiced frames
        voiced_indices = np.where(~np.isnan(pitch))[0]
        
        if len(voiced_indices) == 0:
            # All unvoiced, use default pitch
            return np.full_like(pitch, 200.0)
        
        # Interpolate
        pitch_interp = np.interp(
            np.arange(len(pitch)),
            voiced_indices,
            pitch[voiced_indices]
        )
        
        return pitch_interp
    
    def extract_energy(self, audio):
        """
        Extract energy from audio
        
        Args:
            audio: (time,) - audio waveform
            
        Returns:
            energy: (time // hop_length,) - energy contour
        """
        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        # Compute energy (RMS)
        magnitude = np.abs(stft)
        energy = np.sqrt(np.sum(magnitude ** 2, axis=0))
        
        # Normalize
        energy_log = np.log(energy + 1e-8)
        energy_normalized = (energy_log - energy_log.mean()) / (energy_log.std() + 1e-8)
        
        return torch.from_numpy(energy_normalized).float()
