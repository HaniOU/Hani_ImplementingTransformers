import torch
import torch.nn as nn
import math
from typing import Optional, List
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
        max_len: int = 5000,
        use_rope: bool = False,
        use_swiglu: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_rope = use_rope
        self.use_swiglu = use_swiglu
        
        self.embedding = WordEmbedding(vocab_size, d_model)
        
        if not use_rope:
            self.positional_encoding = PositionalEncoding(d_model, max_len, dropout=dropout)
        else:
            self.positional_encoding = None
        
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            BaseTransformerLayer(
                input_dim=d_model,
                num_heads=n_heads,
                feature_dim=dim_feedforward,
                dropout=dropout,
                use_rope=use_rope,
                max_seq_len=max_len,
                use_swiglu=use_swiglu
            )
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                input_dim=d_model,
                num_heads=n_heads,
                feature_dim=dim_feedforward,
                dropout=dropout,
                use_rope=use_rope,
                max_seq_len=max_len,
                use_swiglu=use_swiglu
            )
            for _ in range(num_decoder_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Weight sharing 
        self.output_projection.weight = self.embedding.embedding.weight
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        x = self.embedding(src) * math.sqrt(self.d_model)
        if self.positional_encoding is not None:
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
        if self.positional_encoding is not None:
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

    @torch.no_grad()
    def generate_greedy(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        bos_idx: int = 1,
        eos_idx: int = 2,
        max_length: int = 100
    ) -> torch.Tensor:
        self.eval()
        
        if src.dim() == 1:
            src = src.unsqueeze(0)
        
        batch_size = src.size(0)
        device = src.device
        
        if src_mask is None:
            src_mask = torch.ones(batch_size, src.size(1), device=device)
        
        encoder_output = self.encode(src, src_mask)
        
        generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            tgt_mask = torch.ones(batch_size, generated.size(1), device=device)
            decoder_output = self.decode(generated, encoder_output, tgt_mask, src_mask)
            
            logits = self.output_projection(decoder_output[:, -1, :])
            
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            finished = finished | (next_token.squeeze(-1) == eos_idx)
            
            if finished.all():
                break
        
        return generated

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        bos_idx: int = 1,
        eos_idx: int = 2,
        pad_idx: int = 0,
        max_length: int = 100,
        beam_size: int = 5,
        length_penalty: float = 0.6
    ) -> torch.Tensor:
        self.eval()
        
        if src.dim() == 1:
            src = src.unsqueeze(0)
        
        batch_size = src.size(0)
        device = src.device
        
        if batch_size > 1:
            results = []
            for i in range(batch_size):
                single_src = src[i:i+1]
                single_mask = src_mask[i:i+1] if src_mask is not None else None
                result = self._beam_search_single(
                    single_src, single_mask, bos_idx, eos_idx,
                    max_length, beam_size, length_penalty
                )
                results.append(result)
            
            max_len = max(r.size(1) for r in results)
            padded = torch.full((batch_size, max_len), pad_idx, dtype=torch.long, device=device)
            for i, r in enumerate(results):
                padded[i, :r.size(1)] = r[0]
            return padded
        
        return self._beam_search_single(
            src, src_mask, bos_idx, eos_idx,
            max_length, beam_size, length_penalty
        )
    
    def _beam_search_single(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        bos_idx: int,
        eos_idx: int,
        max_length: int,
        beam_size: int,
        length_penalty: float
    ) -> torch.Tensor:
        device = src.device
        
        if src_mask is None:
            src_mask = torch.ones(1, src.size(1), device=device)
        
        encoder_output = self.encode(src, src_mask)
        
        encoder_output = encoder_output.repeat(beam_size, 1, 1)
        src_mask = src_mask.repeat(beam_size, 1)
        
        beams = torch.full((beam_size, 1), bos_idx, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = float('-inf')
        
        finished_beams = []
        finished_scores = []
        
        for step in range(max_length - 1):
            tgt_mask = torch.ones(beams.size(0), beams.size(1), device=device)
            decoder_output = self.decode(beams, encoder_output[:beams.size(0)], tgt_mask, src_mask[:beams.size(0)])
            
            logits = self.output_projection(decoder_output[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1)
            
            vocab_size = log_probs.size(-1)
            next_scores = beam_scores.unsqueeze(1) + log_probs
            next_scores = next_scores.view(-1)
            
            num_beams = beams.size(0)
            top_scores, top_indices = next_scores.topk(min(2 * beam_size, next_scores.size(0)), dim=0)
            
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            new_beams = []
            new_scores = []
            
            for score, beam_idx, token_idx in zip(top_scores, beam_indices, token_indices):
                beam_idx = beam_idx.item()
                token_idx = token_idx.item()
                score = score.item()
                
                new_beam = torch.cat([beams[beam_idx:beam_idx+1], torch.tensor([[token_idx]], device=device)], dim=1)
                
                if token_idx == eos_idx:
                    length = new_beam.size(1)
                    normalized_score = score / ((5 + length) / 6) ** length_penalty
                    finished_beams.append(new_beam)
                    finished_scores.append(normalized_score)
                else:
                    new_beams.append(new_beam)
                    new_scores.append(score)
                
                if len(new_beams) >= beam_size:
                    break
            
            if len(new_beams) == 0:
                break
            
            beams = torch.cat(new_beams, dim=0)
            beam_scores = torch.tensor(new_scores, device=device)
            
            if len(finished_beams) >= beam_size:
                break
        
        if len(finished_beams) == 0:
            for i in range(beams.size(0)):
                length = beams[i].size(0)
                normalized_score = beam_scores[i].item() / ((5 + length) / 6) ** length_penalty
                finished_beams.append(beams[i:i+1])
                finished_scores.append(normalized_score)
        
        best_idx = max(range(len(finished_scores)), key=lambda i: finished_scores[i])
        return finished_beams[best_idx]


