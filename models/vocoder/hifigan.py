"""
HiFi-GAN Generator
High-fidelity neural vocoder for mel-spectrogram to waveform conversion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with dilated convolutions"""
    
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 
                     dilation=dilation[0], padding=self.get_padding(kernel_size, dilation[0])),
            nn.Conv1d(channels, channels, kernel_size,
                     dilation=dilation[1], padding=self.get_padding(kernel_size, dilation[1])),
            nn.Conv1d(channels, channels, kernel_size,
                     dilation=dilation[2], padding=self.get_padding(kernel_size, dilation[2]))
        ])
        
        self.convs.apply(self.init_weights)
    
    @staticmethod
    def get_padding(kernel_size, dilation):
        return int((kernel_size * dilation - dilation) / 2)
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        for conv in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = conv(xt)
            x = xt + x
        return x


class MRF(nn.Module):
    """Multi-Receptive Field Fusion (MRF)"""
    
    def __init__(self, channels, resblock_kernel_sizes, resblock_dilation_sizes):
        super().__init__()
        
        self.resblocks = nn.ModuleList()
        for kernel_size, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
            self.resblocks.append(ResBlock(channels, kernel_size, dilation))
    
    def forward(self, x):
        output = None
        for resblock in self.resblocks:
            if output is None:
                output = resblock(x)
            else:
                output = output + resblock(x)
        return output / len(self.resblocks)


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator
    Converts mel-spectrograms to high-quality waveforms
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        
        # Initial convolution
        self.conv_pre = nn.Conv1d(
            config.n_mel_channels,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3
        )
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2 ** i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2
                )
            )
        
        # Multi-receptive field fusion blocks
        self.mrfs = nn.ModuleList()
        for i in range(len(self.ups)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                MRF(channels, config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            )
        
        # Output convolution
        self.conv_post = nn.Conv1d(
            config.upsample_initial_channel // (2 ** len(self.ups)),
            1,
            kernel_size=7,
            stride=1,
            padding=3
        )
        
        # Initialize weights
        self.apply(self.init_weights)
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, mel):
        """
        Generate waveform from mel-spectrogram
        
        Args:
            mel: (batch, n_mel_channels, time) - mel-spectrogram
            
        Returns:
            audio: (batch, 1, time * hop_length) - waveform
        """
        # Transpose if needed (batch, time, n_mel) -> (batch, n_mel, time)
        if mel.size(1) != self.config.n_mel_channels:
            mel = mel.transpose(1, 2)
        
        # Initial convolution
        x = self.conv_pre(mel)
        
        # Upsampling with MRF
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            x = self.mrfs[i](x)
        
        # Output convolution
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    def inference(self, mel):
        """
        Inference mode
        
        Args:
            mel: (batch, time, n_mel) or (time, n_mel) - mel-spectrogram
            
        Returns:
            audio: (batch, time * hop_length) or (time * hop_length,) - waveform
        """
        # Handle single sequence
        single_input = False
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            single_input = True
        
        # Forward pass
        with torch.no_grad():
            audio = self.forward(mel)
        
        # Remove channel dimension
        audio = audio.squeeze(1)
        
        # Remove batch dimension if single input
        if single_input:
            audio = audio.squeeze(0)
        
        return audio
