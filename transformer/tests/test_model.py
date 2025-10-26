"""
Tests for the transformer model.
"""

import pytest
import torch
from transformer.modelling import TransformerModel


def test_model_initialization():
    """Test that the model can be initialized."""
    model = TransformerModel(
        vocab_size=1000,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    )
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_forward():
    """Test the forward pass of the model."""
    vocab_size = 1000
    batch_size = 2
    seq_len = 10
    
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    # Create dummy input
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = model(src, tgt)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, vocab_size)


def test_model_output_range():
    """Test that model outputs are in the expected range."""
    vocab_size = 100
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    src = torch.randint(0, vocab_size, (1, 5))
    tgt = torch.randint(0, vocab_size, (1, 5))
    
    output = model(src, tgt)
    
    # Output should be logits (no softmax applied)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__])

