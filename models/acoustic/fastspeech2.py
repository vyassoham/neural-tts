"""
FastSpeech2 Acoustic Model
Non-autoregressive transformer for mel-spectrogram generation
"""

import torch
import torch.nn as nn
from .transformer import FFTBlock, PositionalEncoding
from .variance_adaptor import VarianceAdaptor


class FastSpeech2(nn.Module):
    """
    FastSpeech2 model for TTS
    Generates mel-spectrograms from phoneme sequences
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Phoneme embedding
        self.phoneme_embedding = nn.Embedding(
            config.vocab_size,
            config.encoder_hidden,
            padding_idx=0
        )
        
        # Speaker embedding (for multi-speaker)
        if config.n_speakers > 1:
            self.speaker_embedding = nn.Embedding(
                config.n_speakers,
                config.speaker_embed_dim
            )
            self.speaker_proj = nn.Linear(
                config.speaker_embed_dim,
                config.encoder_hidden
            )
        else:
            self.speaker_embedding = None
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.encoder_hidden,
            config.max_position_encoding
        )
        
        # Encoder (phoneme -> hidden representation)
        self.encoder = nn.ModuleList([
            FFTBlock(
                hidden_dim=config.encoder_hidden,
                num_heads=config.encoder_heads,
                filter_size=config.encoder_conv1d_filter_size,
                dropout=config.encoder_dropout
            )
            for _ in range(config.encoder_layers)
        ])
        
        # Variance adaptor (duration, pitch, energy)
        self.variance_adaptor = VarianceAdaptor(config)
        
        # Decoder (hidden -> mel-spectrogram)
        self.decoder = nn.ModuleList([
            FFTBlock(
                hidden_dim=config.decoder_hidden,
                num_heads=config.decoder_heads,
                filter_size=config.decoder_conv1d_filter_size,
                dropout=config.decoder_dropout
            )
            for _ in range(config.decoder_layers)
        ])
        
        # Mel-spectrogram projection
        self.mel_linear = nn.Linear(config.decoder_hidden, config.n_mel_channels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, phonemes, speaker_ids=None, src_mask=None,
                durations=None, pitch=None, energy=None,
                duration_control=1.0, pitch_control=1.0, energy_control=1.0,
                max_len=None):
        """
        Forward pass
        
        Args:
            phonemes: (batch, seq_len) - phoneme IDs
            speaker_ids: (batch,) - speaker IDs
            src_mask: (batch, seq_len) - source mask
            durations: (batch, seq_len) - ground truth durations (training)
            pitch: (batch, seq_len) - ground truth pitch (training)
            energy: (batch, seq_len) - ground truth energy (training)
            duration_control: Speed control
            pitch_control: Pitch control
            energy_control: Energy control
            max_len: Maximum output length
            
        Returns:
            mel_output: (batch, max_len, n_mel_channels)
            duration_pred: (batch, seq_len)
            pitch_pred: (batch, seq_len)
            energy_pred: (batch, seq_len)
        """
        # Create source mask if not provided
        if src_mask is None:
            src_mask = (phonemes != 0)  # Assume 0 is padding
        
        # Phoneme embedding
        x = self.phoneme_embedding(phonemes)  # (batch, seq_len, hidden_dim)
        
        # Add speaker embedding if multi-speaker
        if self.speaker_embedding is not None and speaker_ids is not None:
            speaker_embed = self.speaker_embedding(speaker_ids)  # (batch, speaker_dim)
            speaker_embed = self.speaker_proj(speaker_embed)  # (batch, hidden_dim)
            speaker_embed = speaker_embed.unsqueeze(1)  # (batch, 1, hidden_dim)
            x = x + speaker_embed
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Encoder
        for layer in self.encoder:
            x = layer(x, src_mask)
        
        # Variance adaptor
        x, duration_pred, pitch_pred, energy_pred = self.variance_adaptor(
            x, src_mask, durations, pitch, energy,
            duration_control, pitch_control, energy_control, max_len
        )
        
        # Create decoder mask
        dec_seq_len = x.size(1)
        dec_mask = torch.ones(x.size(0), dec_seq_len, dtype=torch.bool, device=x.device)
        
        # Decoder
        for layer in self.decoder:
            x = layer(x, dec_mask)
        
        # Mel-spectrogram projection
        mel_output = self.mel_linear(x)  # (batch, max_len, n_mel_channels)
        
        return mel_output, duration_pred, pitch_pred, energy_pred
    
    def inference(self, phonemes, speaker_id=None, 
                 duration_control=1.0, pitch_control=1.0, energy_control=1.0):
        """
        Inference mode
        
        Args:
            phonemes: (batch, seq_len) or (seq_len,)
            speaker_id: int or (batch,)
            duration_control: Speed control
            pitch_control: Pitch control
            energy_control: Energy control
            
        Returns:
            mel_output: (batch, max_len, n_mel_channels) or (max_len, n_mel_channels)
        """
        # Handle single sequence
        single_input = False
        if phonemes.dim() == 1:
            phonemes = phonemes.unsqueeze(0)
            single_input = True
        
        # Handle speaker ID
        if speaker_id is not None:
            if isinstance(speaker_id, int):
                speaker_id = torch.tensor([speaker_id], device=phonemes.device)
            elif speaker_id.dim() == 0:
                speaker_id = speaker_id.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            mel_output, _, _, _ = self.forward(
                phonemes,
                speaker_ids=speaker_id,
                duration_control=duration_control,
                pitch_control=pitch_control,
                energy_control=energy_control
            )
        
        if single_input:
            mel_output = mel_output.squeeze(0)
        
        return mel_output
