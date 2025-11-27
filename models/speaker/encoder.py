"""
Speaker Encoder
Extracts speaker embeddings from audio for voice cloning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    """
    Speaker Encoder Network
    Extracts speaker embeddings from mel-spectrograms
    Based on GE2E (Generalized End-to-End) architecture
    """
    
    def __init__(self, n_mel_channels=80, embedding_dim=256, lstm_hidden=768, num_layers=3):
        super().__init__()
        
        self.n_mel_channels = n_mel_channels
        self.embedding_dim = embedding_dim
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_mel_channels,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Projection to embedding space
        self.projection = nn.Linear(lstm_hidden, embedding_dim)
        
        # L2 normalization for embeddings
        self.normalize = True
    
    def forward(self, mels):
        """
        Extract speaker embedding from mel-spectrogram
        
        Args:
            mels: (batch, time, n_mel_channels) - mel-spectrograms
            
        Returns:
            embeddings: (batch, embedding_dim) - speaker embeddings
        """
        # LSTM processing
        # mels: (batch, time, n_mel_channels)
        lstm_out, _ = self.lstm(mels)  # (batch, time, lstm_hidden)
        
        # Take the last output
        # Or use average pooling over time
        # Here we use average pooling for better stability
        pooled = torch.mean(lstm_out, dim=1)  # (batch, lstm_hidden)
        
        # Project to embedding space
        embeddings = self.projection(pooled)  # (batch, embedding_dim)
        
        # L2 normalization
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def compute_embedding(self, mel):
        """
        Compute embedding for a single mel-spectrogram
        
        Args:
            mel: (time, n_mel_channels) or (batch, time, n_mel_channels)
            
        Returns:
            embedding: (embedding_dim,) or (batch, embedding_dim)
        """
        # Handle single sequence
        single_input = False
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            single_input = True
        
        # Forward pass
        with torch.no_grad():
            embedding = self.forward(mel)
        
        # Remove batch dimension if single input
        if single_input:
            embedding = embedding.squeeze(0)
        
        return embedding
    
    def similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: (embedding_dim,) or (batch, embedding_dim)
            embedding2: (embedding_dim,) or (batch, embedding_dim)
            
        Returns:
            similarity: scalar or (batch,)
        """
        # Ensure embeddings are normalized
        embedding1 = F.normalize(embedding1, p=2, dim=-1)
        embedding2 = F.normalize(embedding2, p=2, dim=-1)
        
        # Cosine similarity
        similarity = torch.sum(embedding1 * embedding2, dim=-1)
        
        return similarity


class SpeakerEmbeddingTable(nn.Module):
    """
    Learnable speaker embedding table
    Alternative to encoder for known speakers
    """
    
    def __init__(self, num_speakers, embedding_dim=256):
        super().__init__()
        
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim
        
        # Embedding table
        self.embeddings = nn.Embedding(num_speakers, embedding_dim)
        
        # Initialize with normal distribution
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.1)
    
    def forward(self, speaker_ids):
        """
        Get speaker embeddings
        
        Args:
            speaker_ids: (batch,) - speaker IDs
            
        Returns:
            embeddings: (batch, embedding_dim)
        """
        embeddings = self.embeddings(speaker_ids)
        
        # L2 normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
