"""
Training Configuration
Settings for training both acoustic model and vocoder
"""

class TrainingConfig:
    """Configuration for training pipeline"""
    
    # General Training Settings
    seed = 42  # Random seed for reproducibility
    epochs = 1000  # Number of training epochs
    batch_size = 32  # Batch size
    num_workers = 4  # DataLoader workers
    
    # Optimization - Acoustic Model
    acoustic_learning_rate = 1e-4  # Learning rate
    acoustic_weight_decay = 1e-6  # Weight decay
    acoustic_betas = (0.9, 0.98)  # Adam betas
    acoustic_eps = 1e-9  # Adam epsilon
    acoustic_grad_clip_thresh = 1.0  # Gradient clipping threshold
    
    # Optimization - Vocoder
    vocoder_learning_rate = 2e-4  # Learning rate for generator
    vocoder_discriminator_learning_rate = 2e-4  # Learning rate for discriminator
    vocoder_betas = (0.8, 0.99)  # Adam betas
    vocoder_weight_decay = 0.0  # Weight decay
    vocoder_grad_clip_thresh = None  # No gradient clipping for vocoder
    
    # Learning Rate Scheduler
    use_scheduler = True  # Use learning rate scheduler
    scheduler_type = 'noam'  # 'noam', 'exponential', or 'cosine'
    warmup_steps = 4000  # Warmup steps for Noam scheduler
    
    # Loss Weights - Acoustic Model
    mel_loss_weight = 1.0  # Mel-spectrogram loss weight
    duration_loss_weight = 1.0  # Duration predictor loss weight
    pitch_loss_weight = 1.0  # Pitch predictor loss weight
    energy_loss_weight = 1.0  # Energy predictor loss weight
    
    # Loss Weights - Vocoder
    lambda_mel = 45.0  # Mel-spectrogram loss weight
    lambda_feat = 2.0  # Feature matching loss weight
    lambda_adv = 1.0  # Adversarial loss weight
    
    # Checkpointing
    save_interval = 5000  # Save checkpoint every N steps
    eval_interval = 1000  # Evaluate every N steps
    log_interval = 100  # Log every N steps
    keep_checkpoints = 5  # Number of checkpoints to keep
    
    # Mixed Precision Training
    use_fp16 = False  # Use mixed precision training (requires apex or torch.cuda.amp)
    
    # Distributed Training
    distributed = False  # Use distributed training
    world_size = 1  # Number of GPUs
    
    # Data Augmentation
    use_augmentation = True  # Use data augmentation
    pitch_shift_range = 2  # Pitch shift range in semitones
    time_stretch_range = 0.1  # Time stretch range (0.9-1.1)
    
    # Validation
    val_split = 0.05  # Validation split ratio
    
    # Device
    device = 'cuda'  # 'cuda' or 'cpu'
    
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
