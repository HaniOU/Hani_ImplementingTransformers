import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class Attention(nn.Module):

    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
       
        d_k = query.size(-1)
        scores = torch.matmul(query, key.T) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attention_weights, value)
        
        return output




