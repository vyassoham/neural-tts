"""
Training Script for HiFi-GAN Vocoder
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.vocoder_config import VocoderConfig
from config.training_config import TrainingConfig
from models.vocoder.hifigan import HiFiGANGenerator
from models.vocoder.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from data.dataset import VocoderDataset, collate_fn_vocoder
from data.preprocessing import AudioPreprocessor
from training.losses import VocoderLoss
from utils.tools import (
    save_checkpoint, load_checkpoint, count_parameters,
    setup_seed, AverageMeter
)


def train_epoch(generator, mpd, msd, dataloader, criterion, 
                optimizer_g, optimizer_d, preprocessor, device, epoch, writer, config):
    """Train for one epoch"""
    generator.train()
    mpd.train()
    msd.train()
    
    losses_g = AverageMeter()
    losses_d = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for step, batch in enumerate(pbar):
        # Move to device
        mels = batch['mels'].to(device)
        audios = batch['audios'].to(device)
        
        # Add channel dimension to audio
        audios = audios.unsqueeze(1)  # (batch, 1, time)
        
        # ==================== Train Discriminator ====================
        optimizer_d.zero_grad()
        
        # Generate audio
        with torch.no_grad():
            generated_audio = generator(mels)
        
        # Multi-period discriminator
        mpd_real_out, mpd_gen_out, _, _ = mpd(audios, generated_audio.detach())
        mpd_loss = criterion.discriminator_loss(mpd_real_out, mpd_gen_out)
        
        # Multi-scale discriminator
        msd_real_out, msd_gen_out, _, _ = msd(audios, generated_audio.detach())
        msd_loss = criterion.discriminator_loss(msd_real_out, msd_gen_out)
        
        # Total discriminator loss
        d_loss = mpd_loss + msd_loss
        
        d_loss.backward()
        optimizer_d.step()
        
        # ==================== Train Generator ====================
        optimizer_g.zero_grad()
        
        # Generate audio
        generated_audio = generator(mels)
        
        # Compute mel-spectrogram of generated audio
        mel_gen = preprocessor.compute_mel_spectrogram(generated_audio.squeeze(1))
        
        # Multi-period discriminator
        _, mpd_gen_out, mpd_real_feat, mpd_gen_feat = mpd(audios, generated_audio)
        
        # Multi-scale discriminator
        _, msd_gen_out, msd_real_feat, msd_gen_feat = msd(audios, generated_audio)
        
        # Combine discriminator outputs and features
        disc_outputs = mpd_gen_out + msd_gen_out
        real_features = mpd_real_feat + msd_real_feat
        gen_features = mpd_gen_feat + msd_gen_feat
        
        # Generator loss
        g_loss, loss_dict = criterion.generator_total_loss(
            mels, mel_gen, disc_outputs, real_features, gen_features
        )
        
        g_loss.backward()
        optimizer_g.step()
        
        # Update meters
        losses_g.update(g_loss.item())
        losses_d.update(d_loss.item())
        
        # Update progress bar
        pbar.set_postfix({
            'g_loss': f"{losses_g.avg:.4f}",
            'd_loss': f"{losses_d.avg:.4f}"
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + step
        if step % config.log_interval == 0:
            writer.add_scalar('train/generator_loss', losses_g.avg, global_step)
            writer.add_scalar('train/discriminator_loss', losses_d.avg, global_step)
            for key, value in loss_dict.items():
                writer.add_scalar(f'train/{key}', value, global_step)
    
    return losses_g.avg, losses_d.avg


def main():
    parser = argparse.ArgumentParser(description='Train HiFi-GAN Vocoder')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset metadata CSV')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, default='checkpoints/vocoder',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of epochs')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--resume_g', type=str, default=None,
                       help='Path to generator checkpoint to resume from')
    parser.add_argument('--resume_d', type=str, default=None,
                       help='Path to discriminator checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load configs
    vocoder_config = VocoderConfig()
    training_config = TrainingConfig()
    
    # Override with command line args
    training_config.batch_size = args.batch_size
    training_config.epochs = args.epochs
    
    # Setup seed
    setup_seed(training_config.seed)
    
    # Create dataset
    print("Loading dataset...")
    dataset = VocoderDataset(
        metadata_path=args.dataset_path,
        audio_dir=args.audio_dir,
        config=vocoder_config
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=collate_fn_vocoder,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create models
    print("Creating models...")
    generator = HiFiGANGenerator(vocoder_config).to(device)
    mpd = MultiPeriodDiscriminator(vocoder_config.mpd_periods).to(device)
    msd = MultiScaleDiscriminator(
        vocoder_config.msd_num_scales,
        vocoder_config.msd_use_spectral_norm
    ).to(device)
    
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"MPD parameters: {count_parameters(mpd):,}")
    print(f"MSD parameters: {count_parameters(msd):,}")
    
    # Create optimizers
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=training_config.vocoder_learning_rate,
        betas=training_config.vocoder_betas,
        weight_decay=training_config.vocoder_weight_decay
    )
    
    optimizer_d = torch.optim.Adam(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=training_config.vocoder_discriminator_learning_rate,
        betas=training_config.vocoder_betas,
        weight_decay=training_config.vocoder_weight_decay
    )
    
    # Create loss function
    criterion = VocoderLoss(training_config)
    
    # Create preprocessor for mel-spectrogram computation
    preprocessor = AudioPreprocessor(vocoder_config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_g:
        start_epoch, _, _ = load_checkpoint(generator, optimizer_g, args.resume_g, device)
        start_epoch += 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, training_config.epochs):
        avg_g_loss, avg_d_loss = train_epoch(
            generator, mpd, msd, dataloader, criterion,
            optimizer_g, optimizer_d, preprocessor,
            device, epoch, writer, training_config
        )
        
        print(f"Epoch {epoch}: G Loss = {avg_g_loss:.4f}, D Loss = {avg_d_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            g_path = os.path.join(args.output_dir, f'generator_epoch_{epoch+1}.pt')
            d_path = os.path.join(args.output_dir, f'discriminator_epoch_{epoch+1}.pt')
            
            save_checkpoint(generator, optimizer_g, epoch, 0, avg_g_loss, g_path)
            save_checkpoint(mpd, optimizer_d, epoch, 0, avg_d_loss, d_path)
    
    # Save final models
    final_g_path = os.path.join(args.output_dir, 'generator_final.pt')
    final_d_path = os.path.join(args.output_dir, 'discriminator_final.pt')
    
    save_checkpoint(generator, optimizer_g, training_config.epochs, 0, avg_g_loss, final_g_path)
    save_checkpoint(mpd, optimizer_d, training_config.epochs, 0, avg_d_loss, final_d_path)
    
    writer.close()
    print("Training complete!")


if __name__ == '__main__':
    main()
