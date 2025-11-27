"""
Training Script for FastSpeech2 Acoustic Model
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

from config.acoustic_config import AcousticConfig
from config.vocoder_config import VocoderConfig
from config.training_config import TrainingConfig
from models.acoustic.fastspeech2 import FastSpeech2
from data.dataset import TTSDataset, collate_fn_tts
from training.losses import FastSpeech2Loss
from utils.tools import (
    save_checkpoint, load_checkpoint, count_parameters,
    setup_seed, AverageMeter, NoamScheduler, get_mask_from_lengths
)


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, writer, config):
    """Train for one epoch"""
    model.train()
    
    losses = {
        'total': AverageMeter(),
        'mel': AverageMeter(),
        'duration': AverageMeter(),
        'pitch': AverageMeter(),
        'energy': AverageMeter()
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for step, batch in enumerate(pbar):
        # Move to device
        phonemes = batch['phonemes'].to(device)
        mels = batch['mels'].to(device)
        durations = batch['durations'].to(device)
        pitch = batch['pitch'].to(device)
        energy = batch['energy'].to(device)
        speaker_ids = batch['speaker_ids'].to(device)
        phoneme_lengths = batch['phoneme_lengths'].to(device)
        mel_lengths = batch['mel_lengths'].to(device)
        
        # Create masks
        src_mask = get_mask_from_lengths(phoneme_lengths)
        mel_mask = get_mask_from_lengths(mel_lengths)
        
        # Forward pass
        mel_pred, duration_pred, pitch_pred, energy_pred = model(
            phonemes=phonemes,
            speaker_ids=speaker_ids,
            src_mask=src_mask,
            durations=durations,
            pitch=pitch,
            energy=energy,
            max_len=mels.size(2)
        )
        
        # Transpose mel for loss computation
        mel_pred = mel_pred.transpose(1, 2)  # (batch, n_mel, time)
        
        # Compute loss
        predictions = {
            'mel_pred': mel_pred,
            'duration_pred': duration_pred,
            'pitch_pred': pitch_pred,
            'energy_pred': energy_pred
        }
        
        targets = {
            'mel_target': mels,
            'duration_target': durations,
            'pitch_target': pitch,
            'energy_target': energy
        }
        
        masks = {
            'src_mask': src_mask,
            'mel_mask': mel_mask
        }
        
        loss, loss_dict = criterion(predictions, targets, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.acoustic_grad_clip_thresh is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.acoustic_grad_clip_thresh
            )
        
        optimizer.step()
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Update meters
        losses['total'].update(loss_dict['total_loss'])
        losses['mel'].update(loss_dict['mel_loss'])
        losses['duration'].update(loss_dict['duration_loss'])
        losses['pitch'].update(loss_dict['pitch_loss'])
        losses['energy'].update(loss_dict['energy_loss'])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].avg:.4f}",
            'mel': f"{losses['mel'].avg:.4f}"
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + step
        if step % config.log_interval == 0:
            writer.add_scalar('train/total_loss', losses['total'].avg, global_step)
            writer.add_scalar('train/mel_loss', losses['mel'].avg, global_step)
            writer.add_scalar('train/duration_loss', losses['duration'].avg, global_step)
            writer.add_scalar('train/pitch_loss', losses['pitch'].avg, global_step)
            writer.add_scalar('train/energy_loss', losses['energy'].avg, global_step)
    
    return losses['total'].avg


def main():
    parser = argparse.ArgumentParser(description='Train FastSpeech2 Acoustic Model')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset metadata CSV')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, default='checkpoints/acoustic',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of epochs')
    parser.add_argument('--language', type=str, default='en',
                       help='Language code')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load configs
    acoustic_config = AcousticConfig()
    vocoder_config = VocoderConfig()
    training_config = TrainingConfig()
    
    # Override with command line args
    training_config.batch_size = args.batch_size
    training_config.epochs = args.epochs
    
    # Setup seed
    setup_seed(training_config.seed)
    
    # Create dataset
    print("Loading dataset...")
    dataset = TTSDataset(
        metadata_path=args.dataset_path,
        audio_dir=args.audio_dir,
        config=vocoder_config,
        language=args.language
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=collate_fn_tts,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create model
    print("Creating model...")
    model = FastSpeech2(acoustic_config).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.acoustic_learning_rate,
        betas=training_config.acoustic_betas,
        eps=training_config.acoustic_eps,
        weight_decay=training_config.acoustic_weight_decay
    )
    
    # Create scheduler
    if training_config.use_scheduler and training_config.scheduler_type == 'noam':
        scheduler = NoamScheduler(
            optimizer,
            acoustic_config.encoder_hidden,
            training_config.warmup_steps
        )
    else:
        scheduler = None
    
    # Create loss function
    criterion = FastSpeech2Loss(training_config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _, _ = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, training_config.epochs):
        avg_loss = train_epoch(
            model, dataloader, criterion, optimizer, scheduler,
            device, epoch, writer, training_config
        )
        
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f'acoustic_model_epoch_{epoch+1}.pt'
            )
            save_checkpoint(model, optimizer, epoch, 0, avg_loss, checkpoint_path)
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'acoustic_model_final.pt')
    save_checkpoint(model, optimizer, training_config.epochs, 0, avg_loss, final_path)
    
    writer.close()
    print("Training complete!")


if __name__ == '__main__':
    main()
