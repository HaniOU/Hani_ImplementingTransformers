import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class Attention(nn.Module):

    def __init__(self, mask_future: bool = False):
        super().__init__()
        self.mask_future = mask_future
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if self.mask_future:
            seq_len = query.size(-2)  
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device))
            causal_mask = causal_mask.unsqueeze(0)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        output = torch.matmul(attention_weights, value)
        
        return output




