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


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mask_future: bool = False):
        super().__init__()
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.mask_future = mask_future
        
        self.query_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.output_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.attention = Attention(mask_future=mask_future)
        
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
        
        Q = self.query_transform(query)  # (batch, seq_len_q, embedding_dim)
        K = self.key_transform(key)      # (batch, seq_len_k, embedding_dim)
        V = self.value_transform(value)  # (batch, seq_len_v, embedding_dim)
        
        # Reshape for multi-head attention
        # (batch, seq_len, embedding_dim) -> (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Reshape to (batch * num_heads, seq_len, head_dim) for attention
        Q = Q.contiguous().view(batch_size * self.num_heads, seq_len_q, self.head_dim)
        K = K.contiguous().view(batch_size * self.num_heads, seq_len_k, self.head_dim)
        V = V.contiguous().view(batch_size * self.num_heads, seq_len_k, self.head_dim)
        
        if attention_mask is not None:
            # Repeat mask for each head
            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            attention_mask = attention_mask.view(batch_size * self.num_heads, -1)
        
        attention_output = self.attention(Q, K, V, attention_mask)
        # (batch * num_heads, seq_len_q, head_dim)
        
        # Reshape back to (batch, seq_len_q, embedding_dim)
        attention_output = attention_output.view(batch_size, self.num_heads, seq_len_q, self.head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len_q, self.embedding_dim)
        
        output = self.output_transform(attention_output)
        
        return output




