import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class NoamScheduler(_LRScheduler):
    
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
        step = max(1, self.last_epoch + 1)
        
        scale = self.d_model ** (-0.5)
        lr = scale * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        lr = lr * self.factor
        
        return [lr] * self.num_param_groups


class WarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        
        if step < self.warmup_steps:
            warmup_factor = step / self.warmup_steps
        else:
            warmup_factor = 1.0
        
        return [base_lr * warmup_factor for base_lr in self.base_lrs]


def configure_optimizers(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-9
) -> torch.optim.AdamW:

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

    if scheduler_type == 'noam':
        return NoamScheduler(optimizer, **kwargs)
    elif scheduler_type == 'warmup':
        return WarmupScheduler(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
