import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class NoamScheduler(_LRScheduler):
    """
    Formula:
        lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    
    Args:
        optimizer: Wrapped optimizer
        d_model: Model dimension (used for scaling)
        warmup_steps: Number of warmup steps
        factor: Multiplicative factor (default: 1.0)
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        last_epoch: int = -1
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.num_param_groups = len(optimizer.param_groups)
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Returns:
            List of learning rates for each parameter group
        """
        # Step is 1-indexed (self._step_count starts at 1 after first step())
        step = max(1, self.last_epoch + 1)
        
        # Noam formula: d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        scale = self.d_model ** (-0.5)
        lr = scale * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        lr = lr * self.factor
        
        return [lr] * self.num_param_groups


class WarmupScheduler(_LRScheduler):
    """
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Returns:
            List of learning rates for each parameter group
        """
        step = max(1, self.last_epoch + 1)
        
        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / self.warmup_steps
        else:
            # Constant learning rate after warmup
            warmup_factor = 1.0
        
        return [base_lr * warmup_factor for base_lr in self.base_lrs]


def configure_optimizers(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-9
) -> torch.optim.AdamW:
    """
    Args:
        model: The model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam betas (default from the paper: (0.9, 0.98))
        eps: Adam epsilon (default from the paper: 1e-9)
    
    Returns:
        Configured AdamW optimizer
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'bias' in name or 'layer_norm' in name or 'LayerNorm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {
            'params': decay_params,
            'weight_decay': weight_decay
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0
        }
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=betas,
        eps=eps
    )
    
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'noam',
    **kwargs
):
    """
    Args:
        optimizer: The optimizer to wrap
        scheduler_type: Type of scheduler ('noam' or 'warmup')
        **kwargs: Additional arguments for the scheduler
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'noam':
        return NoamScheduler(optimizer, **kwargs)
    elif scheduler_type == 'warmup':
        return WarmupScheduler(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
