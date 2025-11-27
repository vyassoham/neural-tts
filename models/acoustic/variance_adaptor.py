"""
Variance Adaptor for FastSpeech2
Predicts duration, pitch, and energy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariancePredictor(nn.Module):
    """Generic variance predictor (for duration, pitch, energy)"""
    
    def __init__(self, hidden_dim, filter_size, kernel_size, dropout):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, filter_size, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(filter_size, filter_size, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.LayerNorm(filter_size),
                nn.Dropout(dropout)
            )
        ])
        
        self.linear = nn.Linear(filter_size, 1)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len)
        Returns:
            output: (batch, seq_len)
        """
        # Transpose for conv1d
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        
        # Apply conv layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Transpose back
        x = x.transpose(1, 2)  # (batch, seq_len, filter_size)
        
        # Linear projection
        x = self.linear(x).squeeze(-1)  # (batch, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        
        return x


class LengthRegulator(nn.Module):
    """Regulates length based on duration predictions"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, durations, max_len=None):
        """
        Expand hidden states according to durations
        
        Args:
            x: (batch, seq_len, hidden_dim)
            durations: (batch, seq_len) - number of frames for each phoneme
            max_len: Maximum output length
        Returns:
            output: (batch, max_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Calculate output length
        if max_len is None:
            max_len = durations.sum(dim=1).max().item()
        
        # Initialize output
        output = torch.zeros(batch_size, max_len, hidden_dim, 
                           dtype=x.dtype, device=x.device)
        
        # Expand each sequence
        for i in range(batch_size):
            pos = 0
            for j in range(seq_len):
                dur = int(durations[i, j].item())
                if dur > 0 and pos < max_len:
                    end_pos = min(pos + dur, max_len)
                    output[i, pos:end_pos, :] = x[i, j, :]
                    pos = end_pos
                if pos >= max_len:
                    break
        
        return output


class VarianceAdaptor(nn.Module):
    """
    Variance Adaptor module
    Predicts and applies duration, pitch, and energy
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Duration predictor
        self.duration_predictor = VariancePredictor(
            hidden_dim=config.encoder_hidden,
            filter_size=config.duration_predictor_filter_size,
            kernel_size=config.duration_predictor_kernel_size,
            dropout=config.duration_predictor_dropout
        )
        
        # Pitch predictor
        self.pitch_predictor = VariancePredictor(
            hidden_dim=config.encoder_hidden,
            filter_size=config.pitch_predictor_filter_size,
            kernel_size=config.pitch_predictor_kernel_size,
            dropout=config.pitch_predictor_dropout
        )
        
        # Energy predictor
        self.energy_predictor = VariancePredictor(
            hidden_dim=config.encoder_hidden,
            filter_size=config.energy_predictor_filter_size,
            kernel_size=config.energy_predictor_kernel_size,
            dropout=config.energy_predictor_dropout
        )
        
        # Pitch embedding
        self.pitch_bins = nn.Parameter(
            torch.linspace(-1, 1, config.n_pitch_bins - 1),
            requires_grad=False
        )
        self.pitch_embedding = nn.Embedding(config.n_pitch_bins, config.encoder_hidden)
        
        # Energy embedding
        self.energy_bins = nn.Parameter(
            torch.linspace(-1, 1, config.n_energy_bins - 1),
            requires_grad=False
        )
        self.energy_embedding = nn.Embedding(config.n_energy_bins, config.encoder_hidden)
        
        # Length regulator
        self.length_regulator = LengthRegulator()
    
    def get_pitch_embedding(self, pitch, mask=None):
        """Convert continuous pitch to embedding"""
        # Quantize pitch
        pitch_quantized = torch.bucketize(pitch, self.pitch_bins)
        
        # Get embedding
        pitch_embed = self.pitch_embedding(pitch_quantized)
        
        if mask is not None:
            pitch_embed = pitch_embed.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        return pitch_embed
    
    def get_energy_embedding(self, energy, mask=None):
        """Convert continuous energy to embedding"""
        # Quantize energy
        energy_quantized = torch.bucketize(energy, self.energy_bins)
        
        # Get embedding
        energy_embed = self.energy_embedding(energy_quantized)
        
        if mask is not None:
            energy_embed = energy_embed.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        return energy_embed
    
    def forward(self, x, mask=None, durations=None, pitch=None, energy=None,
                duration_control=1.0, pitch_control=1.0, energy_control=1.0, max_len=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim) - encoder output
            mask: (batch, seq_len)
            durations: (batch, seq_len) - ground truth durations (training only)
            pitch: (batch, seq_len) - ground truth pitch (training only)
            energy: (batch, seq_len) - ground truth energy (training only)
            duration_control: Speed control (1.0 = normal)
            pitch_control: Pitch control (1.0 = normal)
            energy_control: Energy control (1.0 = normal)
            max_len: Maximum output length
        Returns:
            output: (batch, max_len, hidden_dim)
            duration_pred: (batch, seq_len)
            pitch_pred: (batch, seq_len)
            energy_pred: (batch, seq_len)
        """
        # Predict duration, pitch, energy
        duration_pred = self.duration_predictor(x, mask)
        pitch_pred = self.pitch_predictor(x, mask)
        energy_pred = self.energy_predictor(x, mask)
        
        # Use ground truth or predictions
        if durations is not None:
            durations_to_use = durations
        else:
            # Convert log-domain predictions to actual durations
            durations_to_use = torch.exp(duration_pred) - 1
            durations_to_use = durations_to_use * duration_control
            durations_to_use = torch.clamp(torch.round(durations_to_use), min=0)
        
        if pitch is not None:
            pitch_to_use = pitch
        else:
            pitch_to_use = pitch_pred * pitch_control
        
        if energy is not None:
            energy_to_use = energy
        else:
            energy_to_use = energy_pred * energy_control
        
        # Add pitch and energy embeddings
        pitch_embed = self.get_pitch_embedding(pitch_to_use, mask)
        energy_embed = self.get_energy_embedding(energy_to_use, mask)
        
        x = x + pitch_embed + energy_embed
        
        # Length regulation
        x = self.length_regulator(x, durations_to_use, max_len)
        
        return x, duration_pred, pitch_pred, energy_pred
