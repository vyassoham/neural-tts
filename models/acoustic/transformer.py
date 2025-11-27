"""
Transformer Blocks for FastSpeech2
Multi-head self-attention with feed-forward networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) - True for valid positions
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x)  # (batch, seq_len, hidden_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, head_dim)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch, num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for heads
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)
        
        # Final linear projection
        output = self.out(output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, hidden_dim, filter_size, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, filter_size)
        self.w2 = nn.Linear(filter_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        x = self.w1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class FFTBlock(nn.Module):
    """
    Feed-Forward Transformer Block
    Multi-head attention + position-wise feed-forward
    """
    
    def __init__(self, hidden_dim, num_heads, filter_size, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Position-wise feed-forward
        self.ffn = PositionwiseFeedForward(hidden_dim, filter_size, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len)
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        # Multi-head attention with residual connection
        residual = x
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = self.attn_norm(residual + x)
        
        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.ffn_norm(residual + x)
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, hidden_dim, max_len=10000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                            (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, hidden_dim)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
