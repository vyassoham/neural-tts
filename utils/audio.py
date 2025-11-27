"""
Audio processing utilities
"""

import torch
import numpy as np


class AudioProcessor:
    """Utility class for audio processing"""
    
    @staticmethod
    def dynamic_range_compression(x, C=1, clip_val=1e-5):
        """
        Dynamic range compression for audio signals
        
        Args:
            x: Input signal
            C: Compression factor
            clip_val: Minimum value for clipping
            
        Returns:
            Compressed signal
        """
        return torch.log(torch.clamp(x, min=clip_val) * C)
    
    @staticmethod
    def dynamic_range_decompression(x, C=1):
        """
        Dynamic range decompression
        
        Args:
            x: Compressed signal
            C: Compression factor
            
        Returns:
            Decompressed signal
        """
        return torch.exp(x) / C
    
    @staticmethod
    def griffin_lim(magnitudes, n_iters=32, n_fft=1024, hop_length=256, win_length=1024):
        """
        Griffin-Lim algorithm for phase reconstruction
        
        Args:
            magnitudes: (n_fft // 2 + 1, time) - magnitude spectrogram
            n_iters: Number of iterations
            n_fft: FFT size
            hop_length: Hop length
            win_length: Window length
            
        Returns:
            audio: (time,) - reconstructed audio
        """
        # Convert to numpy if tensor
        if isinstance(magnitudes, torch.Tensor):
            magnitudes = magnitudes.cpu().numpy()
        
        # Initialize with random phase
        angles = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
        
        # Iterative phase reconstruction
        for _ in range(n_iters):
            # Inverse STFT
            full_complex = magnitudes * angles
            audio = librosa.istft(full_complex, hop_length=hop_length, 
                                win_length=win_length, window='hann')
            
            # Forward STFT
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, window='hann')
            
            # Update phase
            angles = np.exp(1j * np.angle(stft))
        
        # Final reconstruction
        audio = librosa.istft(magnitudes * angles, hop_length=hop_length,
                            win_length=win_length, window='hann')
        
        return audio
