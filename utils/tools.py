"""
General utility functions
"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np


def get_mask_from_lengths(lengths, max_len=None):
    """
    Create mask from lengths
    
    Args:
        lengths: (batch,) - sequence lengths
        max_len: Maximum length (None = use max from lengths)
        
    Returns:
        mask: (batch, max_len) - boolean mask (True for valid positions)
    """
    if max_len is None:
        max_len = torch.max(lengths).item()
    
    batch_size = lengths.size(0)
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)
    
    return mask


def plot_spectrogram(spectrogram, title="Spectrogram", save_path=None):
    """
    Plot spectrogram
    
    Args:
        spectrogram: (n_mel, time) or (time, n_mel) - spectrogram
        title: Plot title
        save_path: Path to save plot (None = display)
    """
    # Convert to numpy if tensor
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.cpu().numpy()
    
    # Ensure correct orientation
    if spectrogram.shape[0] > spectrogram.shape[1]:
        spectrogram = spectrogram.T
    
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_checkpoint(model, optimizer, epoch, step, loss, path):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        path: Save path
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss
    }
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path, device='cpu'):
    """
    Load model checkpoint
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        path: Checkpoint path
        device: Device to load to
        
    Returns:
        epoch: Loaded epoch
        step: Loaded step
        loss: Loaded loss
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', 0.0)
    
    print(f"Checkpoint loaded: {path}")
    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss:.4f}")
    
    return epoch, step, loss


def count_parameters(model):
    """
    Count trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        count: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    """
    Setup random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_learning_rate(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class NoamScheduler:
    """Noam learning rate scheduler"""
    
    def __init__(self, optimizer, hidden_dim, warmup_steps=4000):
        self.optimizer = optimizer
        self.hidden_dim = hidden_dim
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        return self.hidden_dim ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
