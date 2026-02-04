import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .positional_encoding import RotaryPositionalEmbedding


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


class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        num_heads: int, 
        mask_future: bool = False,
        use_rope: bool = False,
        max_seq_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.mask_future = mask_future
        self.use_rope = use_rope
        
        self.query_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.output_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        Q = self.query_transform(query)
        K = self.key_transform(key)
        V = self.value_transform(value)
            
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.use_rope:
            Q, K = self.rope(Q, K, seq_len=max(seq_len_q, seq_len_k))
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if self.mask_future:
            causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=query.device))
            scores = scores.masked_fill(causal_mask == 0, -1e4)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, -1e4)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len_q, self.embedding_dim)
        
        output = self.output_transform(attention_output)
        
        return output




