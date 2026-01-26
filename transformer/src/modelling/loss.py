"""
Loss functions for transformer training.
"""
import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Label smoothing helps prevent the model from becoming overconfident
    by distributing some probability mass to incorrect labels.
    
    Args:
        smoothing: Label smoothing factor (typically 0.1)
        ignore_index: Index to ignore in loss calculation (for padding)
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (batch_size, seq_len, vocab_size) or (batch_size * seq_len, vocab_size)
            target: Targets (batch_size, seq_len) or (batch_size * seq_len,)
        
        Returns:
            Loss value
        """
        # Reshape if needed
        if pred.dim() == 3:
            pred = pred.reshape(-1, pred.size(-1))
        if target.dim() == 2:
            target = target.reshape(-1)
        
        vocab_size = pred.size(-1)
        
        # Create smoothed target distribution
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (vocab_size - 1))
        
        # Set the true label probability
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Mask out ignored indices
        mask = (target == self.ignore_index)
        if mask.any():
            true_dist[mask] = 0
        
        # Calculate KL divergence
        pred_log_prob = torch.nn.functional.log_softmax(pred, dim=-1)
        loss = -(true_dist * pred_log_prob).sum(dim=-1)
        
        # Apply mask
        if mask.any():
            loss = loss.masked_fill(mask, 0)
            return loss.sum() / (~mask).sum()
        
        return loss.mean()


def get_loss_function(loss_type: str = 'cross_entropy', **kwargs):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('cross_entropy' or 'label_smoothing')
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function
    """
    if loss_type == 'cross_entropy':
        ignore_index = kwargs.get('ignore_index', -100)
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        ignore_index = kwargs.get('ignore_index', -100)
        return LabelSmoothingCrossEntropy(smoothing=smoothing, ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
