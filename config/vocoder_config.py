"""
HiFi-GAN Vocoder Configuration
Optimized for ultra-high clarity and fast inference
"""

class VocoderConfig:
    """Configuration for HiFi-GAN vocoder"""
    
    # Audio Settings
    sampling_rate = 22050  # Audio sampling rate (Hz)
    n_fft = 1024  # FFT size
    hop_length = 256  # Hop length for STFT
    win_length = 1024  # Window length for STFT
    n_mel_channels = 80  # Number of mel channels (must match acoustic model)
    mel_fmin = 0.0  # Minimum frequency for mel
    mel_fmax = 8000.0  # Maximum frequency for mel (None = sampling_rate/2)
    
    # Generator Architecture
    resblock_kernel_sizes = [3, 7, 11]  # Residual block kernel sizes
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]  # Dilation rates
    upsample_rates = [8, 8, 2, 2]  # Upsampling rates (product should equal hop_length)
    upsample_initial_channel = 512  # Initial channels in generator
    upsample_kernel_sizes = [16, 16, 4, 4]  # Upsampling kernel sizes
    
    # Discriminator Architecture
    # Multi-Period Discriminator
    mpd_periods = [2, 3, 5, 7, 11]  # Periods for multi-period discriminator
    
    # Multi-Scale Discriminator
    msd_num_scales = 3  # Number of scales
    msd_use_spectral_norm = False  # Use spectral normalization
    
    # Training
    segment_size = 8192  # Audio segment size for training
    
    # Model Paths
    pretrained_path = None  # Path to pretrained model
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
