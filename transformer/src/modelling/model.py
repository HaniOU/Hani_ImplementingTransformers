import torch
import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, x):
        """
        x: Tensor of shape (..., ) with token ids
        returns: Tensor of shape (..., embedding_dim)
        """
        return self.embedding(x)


