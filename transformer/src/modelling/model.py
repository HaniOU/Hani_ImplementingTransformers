import torch
import torch.nn as nn
import math
from typing import Optional
from .positional_encoding import PositionalEncoding
from .functional import BaseTransformerLayer, TransformerDecoderLayer


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, x):
     
        return self.embedding(x)


class TransformerModel(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = WordEmbedding(vocab_size, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            BaseTransformerLayer(
                input_dim=d_model,
                num_heads=n_heads,
                feature_dim=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                input_dim=d_model,
                num_heads=n_heads,
                feature_dim=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_decoder_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Weight sharing 
        self.output_projection.weight = self.embedding.embedding.weight
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source sequence (batch_size, src_seq_len)
            src_mask: Source mask (batch_size, src_seq_len)
            
        Returns:
            Encoder output (batch_size, src_seq_len, d_model)
        """

        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask=src_mask)
        
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(
                x,
                encoder_output,
                encoder_attention_mask=src_mask,
                attention_mask=tgt_mask
            )
        
        return x
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:


        encoder_output = self.encode(src, src_mask)
        
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        output = self.output_projection(decoder_output)
        
        return output


