"""
Utilities Module for Dynamic Flow Networks

This module contains utility functions and classes for DFN development:
- Training utilities
- Data processing helpers
- Model configuration
- Performance monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information in MB"""
    if torch.cuda.is_available():
        # Current GPU memory allocated
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB

        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'total_mb': total_memory,
            'free_mb': total_memory - allocated,
            'utilization_percent': (allocated / total_memory) * 100 if total_memory > 0 else 0
        }
    else:
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'total_mb': 0.0,
            'free_mb': 0.0,
            'utilization_percent': 0.0
        }


def print_gpu_memory_info(label: str = "") -> None:
    """Print GPU memory information with optional label"""
    if torch.cuda.is_available():
        info = get_gpu_memory_info()
        if label:
            print(f"{label}")
        print(f"   GPU Memory: {info['allocated_mb']:.1f}MB / {info['total_mb']:.1f}MB ({info['utilization_percent']:.1f}%)")
        print(f"   Reserved: {info['reserved_mb']:.1f}MB, Free: {info['free_mb']:.1f}MB")
    else:
        print(f"{label}Running on CPU - No GPU memory info available")


def print_memory_summary(model: nn.Module, label: str = "Memory Summary") -> None:
    """Print comprehensive memory information"""
    print(f"\n{label}")
    print("=" * 40)

    # Model memory footprint
    memory_info = get_model_memory_footprint(model)
    print(f"📊 Model Memory Footprint: {memory_info['total_mb']:.1f}MB")
    print(f"   Parameters: {memory_info['parameters_mb']:.1f}MB")
    print(f"   Buffers: {memory_info['buffers_mb']:.1f}MB")

    # GPU/CPU information
    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        print(f"🎮 GPU Memory: {gpu_info['allocated_mb']:.1f}MB / {gpu_info['total_mb']:.1f}MB ({gpu_info['utilization_percent']:.1f}%)")
        print(f"   Peak VRAM Usage: {gpu_info['reserved_mb']:.1f}MB")
        print(f"   VRAM Efficiency: {gpu_info['utilization_percent']:.1f}%")
    else:
        print("💻 Running on CPU")
        print("   System RAM: Model parameters loaded in system memory")

        # Show device information
        device = next(model.parameters()).device
        print(f"   Active Device: {device}")

    print(f"🔢 Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📏 Model Size: {memory_info['total_mb']:.1f}MB")


def get_model_memory_footprint(model: nn.Module) -> Dict[str, float]:
    """Calculate model memory footprint in MB"""
    param_size = 0
    buffer_size = 0

    # Parameters
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    # Buffers (non-parameter tensors like running means in batch norm)
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_mb = (param_size + buffer_size) / 1024 / 1024

    return {
        'parameters_mb': param_size / 1024 / 1024,
        'buffers_mb': buffer_size / 1024 / 1024,
        'total_mb': total_size_mb
    }


class DFNConfig:
    """Configuration class for DFN models"""

    def __init__(self,
                 vocab_size: int = 1000,
                 max_seq_len: int = 256,
                 dim: int = 512,
                 field_dim: int = 256,
                 num_entities: int = 128,
                 num_layers: int = 6,
                 k: int = 16,
                 alpha: float = 0.1,
                 sigma: float = 1.0,
                 memory_size: int = 64,
                 beta: float = 0.9,
                 dropout: float = 0.1,
                 pos_dim: int = 64,
                 **kwargs):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.field_dim = field_dim
        self.num_entities = num_entities
        self.num_layers = num_layers
        self.k = k
        self.alpha = alpha
        self.sigma = sigma
        self.memory_size = memory_size
        self.beta = beta
        self.dropout = dropout
        self.pos_dim = pos_dim

        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'dim': self.dim,
            'field_dim': self.field_dim,
            'num_entities': self.num_entities,
            'num_layers': self.num_layers,
            'k': self.k,
            'alpha': self.alpha,
            'sigma': self.sigma,
            'memory_size': self.memory_size,
            'beta': self.beta,
            'dropout': self.dropout,
            'pos_dim': self.pos_dim
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DFNConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


def create_synthetic_text_data(vocab_size: int = 1000, seq_len: int = 128,
                              batch_size: int = 4, num_batches: int = 100):
    """Create synthetic text data for training"""
    for _ in range(num_batches):
        # Generate random tokens as training data
        x = torch.randint(1, vocab_size, (batch_size, seq_len))  # Avoid token 0 (padding)

        # Target is the next token (shifted)
        y = torch.roll(x, shifts=-1, dims=1)
        y[:, -1] = 0  # Padding in the last position

        yield x, y


def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               optimizer: torch.optim.Optimizer, criterion,
               scheduler: Optional[Any] = None) -> float:
    """Single training step"""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    logits = model(x)  # [batch, seq_len, vocab_size]

    # Calculate loss (cross-entropy)
    loss = criterion(logits.reshape(-1, model.vocab_size), y.view(-1))

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return loss.item()


def compute_perplexity(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute perplexity for language model evaluation"""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.view(-1))
        perplexity = torch.exp(loss)
    return perplexity.item()


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


class LearningRateScheduler:
    """Custom learning rate scheduler for DFN training"""

    def __init__(self, optimizer, warmup_steps: int = 1000, max_steps: int = 10000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step_count = 0

    def step(self):
        """Update learning rate"""
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


# Export all functions and classes
__all__ = [
    # Memory monitoring functions
    'get_gpu_memory_info',
    'print_gpu_memory_info',
    'print_memory_summary',
    'get_model_memory_footprint',
    'clear_gpu_cache',

    # Configuration
    'DFNConfig',

    # Training utilities
    'create_synthetic_text_data',
    'train_step',
    'compute_perplexity',
    'count_parameters',
    'get_model_size_mb',

    # Learning rate scheduler
    'LearningRateScheduler'
]
