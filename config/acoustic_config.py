"""
FastSpeech2 Acoustic Model Configuration
Optimized for clarity, speed, and multilingual support
"""

class AcousticConfig:
    """Configuration for FastSpeech2 acoustic model"""
    
    # Model Architecture
    vocab_size = 512  # Phoneme vocabulary size
    max_seq_len = 1000  # Maximum sequence length
    
    # Encoder/Decoder
    encoder_hidden = 256  # Hidden dimension
    encoder_layers = 4  # Number of encoder layers
    encoder_heads = 2  # Multi-head attention heads
    encoder_conv1d_filter_size = 1024  # Conv1D filter size
    encoder_conv1d_kernel_size = 9  # Conv1D kernel size
    encoder_dropout = 0.2  # Dropout rate
    
    decoder_hidden = 256  # Decoder hidden dimension
    decoder_layers = 4  # Number of decoder layers
    decoder_heads = 2  # Decoder attention heads
    decoder_conv1d_filter_size = 1024
    decoder_conv1d_kernel_size = 9
    decoder_dropout = 0.2
    
    # Variance Adaptors
    variance_predictor_filter_size = 256
    variance_predictor_kernel_size = 3
    variance_predictor_dropout = 0.5
    
    # Duration Predictor
    duration_predictor_filter_size = 256
    duration_predictor_kernel_size = 3
    duration_predictor_dropout = 0.5
    
    # Pitch Predictor
    pitch_feature_level = "phoneme_level"  # or "frame_level"
    pitch_predictor_filter_size = 256
    pitch_predictor_kernel_size = 5
    pitch_predictor_dropout = 0.5
    n_pitch_bins = 256  # Quantization bins for pitch
    
    # Energy Predictor
    energy_feature_level = "phoneme_level"  # or "frame_level"
    energy_predictor_filter_size = 256
    energy_predictor_kernel_size = 5
    energy_predictor_dropout = 0.5
    n_energy_bins = 256  # Quantization bins for energy
    
    # Multi-speaker
    n_speakers = 1  # Number of speakers (set to >1 for multi-speaker)
    speaker_embed_dim = 256  # Speaker embedding dimension
    
    # Mel-spectrogram
    n_mel_channels = 80  # Number of mel channels
    
    # Positional Encoding
    max_position_encoding = 10000
    
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
