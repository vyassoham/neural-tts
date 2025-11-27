"""
Loss Functions for TTS Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSpeech2Loss(nn.Module):
    """Loss function for FastSpeech2 acoustic model"""
    
    def __init__(self, config):
        super().__init__()
        self.mel_loss_weight = config.mel_loss_weight
        self.duration_loss_weight = config.duration_loss_weight
        self.pitch_loss_weight = config.pitch_loss_weight
        self.energy_loss_weight = config.energy_loss_weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def forward(self, predictions, targets, masks=None):
        """
        Compute loss
        
        Args:
            predictions: dict with keys:
                - mel_pred: (batch, max_len, n_mel)
                - duration_pred: (batch, seq_len)
                - pitch_pred: (batch, seq_len)
                - energy_pred: (batch, seq_len)
            targets: dict with keys:
                - mel_target: (batch, max_len, n_mel)
                - duration_target: (batch, seq_len)
                - pitch_target: (batch, seq_len)
                - energy_target: (batch, seq_len)
            masks: dict with keys:
                - src_mask: (batch, seq_len)
                - mel_mask: (batch, max_len)
                
        Returns:
            total_loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        # Mel-spectrogram loss
        mel_pred = predictions['mel_pred']
        mel_target = targets['mel_target']
        
        if masks and 'mel_mask' in masks:
            mel_mask = masks['mel_mask'].unsqueeze(-1)  # (batch, max_len, 1)
            mel_pred = mel_pred.masked_select(mel_mask)
            mel_target = mel_target.masked_select(mel_mask)
        
        mel_loss = self.mae_loss(mel_pred, mel_target)
        
        # Duration loss (log domain)
        duration_pred = predictions['duration_pred']
        duration_target = targets['duration_target']
        
        # Convert to log domain
        duration_target_log = torch.log(duration_target.float() + 1)
        
        if masks and 'src_mask' in masks:
            src_mask = masks['src_mask']
            duration_pred = duration_pred.masked_select(src_mask)
            duration_target_log = duration_target_log.masked_select(src_mask)
        
        duration_loss = self.mse_loss(duration_pred, duration_target_log)
        
        # Pitch loss
        pitch_pred = predictions['pitch_pred']
        pitch_target = targets['pitch_target']
        
        if masks and 'src_mask' in masks:
            pitch_pred = pitch_pred.masked_select(src_mask)
            pitch_target = pitch_target.masked_select(src_mask)
        
        pitch_loss = self.mse_loss(pitch_pred, pitch_target)
        
        # Energy loss
        energy_pred = predictions['energy_pred']
        energy_target = targets['energy_target']
        
        if masks and 'src_mask' in masks:
            energy_pred = energy_pred.masked_select(src_mask)
            energy_target = energy_target.masked_select(src_mask)
        
        energy_loss = self.mse_loss(energy_pred, energy_target)
        
        # Total loss
        total_loss = (
            self.mel_loss_weight * mel_loss +
            self.duration_loss_weight * duration_loss +
            self.pitch_loss_weight * pitch_loss +
            self.energy_loss_weight * energy_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'mel_loss': mel_loss.item(),
            'duration_loss': duration_loss.item(),
            'pitch_loss': pitch_loss.item(),
            'energy_loss': energy_loss.item()
        }
        
        return total_loss, loss_dict


class VocoderLoss(nn.Module):
    """Loss function for HiFi-GAN vocoder"""
    
    def __init__(self, config):
        super().__init__()
        self.lambda_mel = config.lambda_mel
        self.lambda_feat = config.lambda_feat
        self.lambda_adv = config.lambda_adv
    
    def generator_loss(self, disc_outputs):
        """
        Generator adversarial loss
        
        Args:
            disc_outputs: List of discriminator outputs for generated audio
            
        Returns:
            loss: Generator adversarial loss
        """
        loss = 0
        for disc_out in disc_outputs:
            # Least squares GAN loss
            loss += torch.mean((disc_out - 1) ** 2)
        
        return loss / len(disc_outputs)
    
    def discriminator_loss(self, real_outputs, gen_outputs):
        """
        Discriminator adversarial loss
        
        Args:
            real_outputs: List of discriminator outputs for real audio
            gen_outputs: List of discriminator outputs for generated audio
            
        Returns:
            loss: Discriminator adversarial loss
        """
        loss = 0
        for real_out, gen_out in zip(real_outputs, gen_outputs):
            # Least squares GAN loss
            real_loss = torch.mean((real_out - 1) ** 2)
            gen_loss = torch.mean(gen_out ** 2)
            loss += real_loss + gen_loss
        
        return loss / len(real_outputs)
    
    def feature_matching_loss(self, real_features, gen_features):
        """
        Feature matching loss
        
        Args:
            real_features: List of feature maps from real audio
            gen_features: List of feature maps from generated audio
            
        Returns:
            loss: Feature matching loss
        """
        loss = 0
        count = 0
        
        for real_feat_list, gen_feat_list in zip(real_features, gen_features):
            for real_feat, gen_feat in zip(real_feat_list, gen_feat_list):
                loss += F.l1_loss(gen_feat, real_feat.detach())
                count += 1
        
        return loss / count if count > 0 else loss
    
    def mel_spectrogram_loss(self, mel_real, mel_gen):
        """
        Mel-spectrogram reconstruction loss
        
        Args:
            mel_real: (batch, n_mel, time) - real mel-spectrogram
            mel_gen: (batch, n_mel, time) - generated mel-spectrogram
            
        Returns:
            loss: Mel-spectrogram loss
        """
        return F.l1_loss(mel_gen, mel_real)
    
    def generator_total_loss(self, mel_real, mel_gen, disc_outputs, 
                            real_features, gen_features):
        """
        Total generator loss
        
        Args:
            mel_real: Real mel-spectrogram
            mel_gen: Generated mel-spectrogram
            disc_outputs: Discriminator outputs for generated audio
            real_features: Feature maps from real audio
            gen_features: Feature maps from generated audio
            
        Returns:
            total_loss: Total generator loss
            loss_dict: Dictionary of individual losses
        """
        # Adversarial loss
        adv_loss = self.generator_loss(disc_outputs)
        
        # Feature matching loss
        feat_loss = self.feature_matching_loss(real_features, gen_features)
        
        # Mel-spectrogram loss
        mel_loss = self.mel_spectrogram_loss(mel_real, mel_gen)
        
        # Total loss
        total_loss = (
            self.lambda_adv * adv_loss +
            self.lambda_feat * feat_loss +
            self.lambda_mel * mel_loss
        )
        
        loss_dict = {
            'gen_total_loss': total_loss.item(),
            'gen_adv_loss': adv_loss.item(),
            'gen_feat_loss': feat_loss.item(),
            'gen_mel_loss': mel_loss.item()
        }
        
        return total_loss, loss_dict
