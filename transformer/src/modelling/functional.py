import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, feature_dim)
        self.linear2 = nn.Linear(feature_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


class BaseTransformerLayer(nn.Module): 
    def __init__(
        self, 
        input_dim: int, 
        num_heads: int, 
        feature_dim: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(
            embedding_dim=input_dim,
            num_heads=num_heads,
            mask_future=False
        )
        
        self.feature_transformation = PositionWiseFeedForward(
            input_dim=input_dim,
            feature_dim=feature_dim
        )
        
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Sublayer 1: Multi-Head Self-Attention + Residual + LayerNorm
        attention_output = self.self_attention(x, x, x, attention_mask)
        attention_output = self.dropout(attention_output)
        x = self.layer_norm_1(x + attention_output)
        
        # Sublayer 2: Feed Forward + Residual + LayerNorm
        ff_output = self.feature_transformation(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm_2(x + ff_output)
        
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        feature_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(
            embedding_dim=input_dim,
            num_heads=num_heads,
            mask_future=True
        )
        
        self.encoder_attention = MultiHeadAttention(
            embedding_dim=input_dim,
            num_heads=num_heads,
            mask_future=False
        )
        
        self.feature_transformation = PositionWiseFeedForward(
            input_dim=input_dim,
            feature_dim=feature_dim
        )
        
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Sublayer 1: Masked Multi-Head Self-Attention + Residual + LayerNorm
        self_attention_output = self.self_attention(x, x, x, attention_mask)
        self_attention_output = self.dropout(self_attention_output)
        x = self.layer_norm_1(x + self_attention_output)
        
        # Sublayer 2: Multi-Head Encoder Cross-Attention + Residual + LayerNorm
        # Query from decoder, Key and Value from encoder
        cross_attention_output = self.encoder_attention(x, encoder, encoder, encoder_attention_mask)
        cross_attention_output = self.dropout(cross_attention_output)
        x = self.layer_norm_2(x + cross_attention_output)
        
        # Sublayer 3: Feed Forward + Residual + LayerNorm
        ff_output = self.feature_transformation(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm_3(x + ff_output)
        
        return x

