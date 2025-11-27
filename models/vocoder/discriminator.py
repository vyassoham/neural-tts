"""
Discriminators for HiFi-GAN
Multi-Period and Multi-Scale discriminators for adversarial training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodDiscriminator(nn.Module):
    """Discriminator for a specific period"""
    
    def __init__(self, period):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
        ])
        
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, time) - audio waveform
        Returns:
            output: (batch, 1, ...)
            feature_maps: List of intermediate features
        """
        feature_maps = []
        
        # Reshape to 2D with period
        batch, channels, time = x.shape
        if time % self.period != 0:
            # Pad to make divisible by period
            pad_amount = self.period - (time % self.period)
            x = F.pad(x, (0, pad_amount), mode='reflect')
            time = time + pad_amount
        
        x = x.view(batch, channels, time // self.period, self.period)
        
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        
        # Output layer
        x = self.conv_post(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD)"""
    
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in periods
        ])
    
    def forward(self, real_audio, generated_audio):
        """
        Args:
            real_audio: (batch, 1, time)
            generated_audio: (batch, 1, time)
        Returns:
            real_outputs: List of discriminator outputs for real audio
            gen_outputs: List of discriminator outputs for generated audio
            real_feature_maps: List of feature maps for real audio
            gen_feature_maps: List of feature maps for generated audio
        """
        real_outputs = []
        gen_outputs = []
        real_feature_maps = []
        gen_feature_maps = []
        
        for disc in self.discriminators:
            real_out, real_feat = disc(real_audio)
            gen_out, gen_feat = disc(generated_audio)
            
            real_outputs.append(real_out)
            gen_outputs.append(gen_out)
            real_feature_maps.append(real_feat)
            gen_feature_maps.append(gen_feat)
        
        return real_outputs, gen_outputs, real_feature_maps, gen_feature_maps


class ScaleDiscriminator(nn.Module):
    """Discriminator for a specific scale"""
    
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        
        norm_f = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, time)
        Returns:
            output: (batch, 1, ...)
            feature_maps: List of intermediate features
        """
        feature_maps = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        
        x = self.conv_post(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, feature_maps


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator (MSD)"""
    
    def __init__(self, num_scales=3, use_spectral_norm=False):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm) for _ in range(num_scales)
        ])
        
        # Pooling layers for different scales
        self.poolings = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])
    
    def forward(self, real_audio, generated_audio):
        """
        Args:
            real_audio: (batch, 1, time)
            generated_audio: (batch, 1, time)
        Returns:
            real_outputs: List of discriminator outputs for real audio
            gen_outputs: List of discriminator outputs for generated audio
            real_feature_maps: List of feature maps for real audio
            gen_feature_maps: List of feature maps for generated audio
        """
        real_outputs = []
        gen_outputs = []
        real_feature_maps = []
        gen_feature_maps = []
        
        for i, disc in enumerate(self.discriminators):
            # Apply discriminator at current scale
            real_out, real_feat = disc(real_audio)
            gen_out, gen_feat = disc(generated_audio)
            
            real_outputs.append(real_out)
            gen_outputs.append(gen_out)
            real_feature_maps.append(real_feat)
            gen_feature_maps.append(gen_feat)
            
            # Downsample for next scale
            if i < len(self.poolings):
                real_audio = self.poolings[i](real_audio)
                generated_audio = self.poolings[i](generated_audio)
        
        return real_outputs, gen_outputs, real_feature_maps, gen_feature_maps
