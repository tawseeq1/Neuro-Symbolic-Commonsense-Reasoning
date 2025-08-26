"""
Optimizer and scheduler utilities for training.
"""

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR
from typing import Dict, Any


def get_optimizer(model, type: str = "adamw", lr: float = 2e-5, 
                 weight_decay: float = 0.01, **kwargs) -> torch.optim.Optimizer:
    """Get optimizer for training."""
    if type.lower() == "adamw":
        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif type.lower() == "adam":
        return Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif type.lower() == "sgd":
        return SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {type}")


def get_scheduler(optimizer, type: str = "cosine", warmup_steps: int = 0,
                 num_training_steps: int = 1000, **kwargs):
    """Get learning rate scheduler."""
    if type.lower() == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            **kwargs
        )
    elif type.lower() == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_training_steps,
            **kwargs
        )
    elif type.lower() == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 100),
            gamma=kwargs.get("gamma", 0.1),
            **kwargs
        )
    elif type.lower() == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {type}")


def get_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Get warmup scheduler."""
    if warmup_steps == 0:
        return None
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) 