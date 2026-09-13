"""Token and position embedding composition."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from minibert.model.components import LayerNorm, TokenEmbedding


class PositionEmbedding(Module):
    """Learned position embeddings."""

    def __init__(
        self,
        max_positions: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        if max_positions < 1:
            raise ValueError("max_positions must be positive")
        
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        
        self.max_positions = max_positions
        self.hidden_size = hidden_size

        self.weight = Parameter(
            torch.randn(max_positions, hidden_size)
            * (1.0 / math.sqrt(hidden_size))
        )
    
    def forward(self, position_ids: Tensor) -> Tensor:
        if position_ids.dtype != torch.long:
            raise TypeError("position_ids must use torch.long")
        
        if torch.any(position_ids < 0):
            raise ValueError("position IDs must be non-negative")

        if torch.any(position_ids >= self.max_positions):
            raise ValueError("position IDs exceed max_positions")
        
        return self.weight[position_ids]

class BertEmbeddings(Module):
    """Token plus position embeddings followed by LayerNorm."""

    def __init__(
        self,
        vocab_size: int,
        max_positions: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )

        self.position_embedding = PositionEmbedding(
            max_positions=max_positions,
            hidden_size=hidden_size,
        )

        self.layer_norm = LayerNorm(hidden_size)
    
    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be shape [batch, sequence]")
        
        batch_size, sequence_length = input_ids.shape

        token_vectors = self.token_embedding(input_ids)

        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        )

        position_ids = position_ids.unsqueeze(0).expand(
            batch_size,
            sequence_length,
        )

        position_vectors = self.position_embedding(position_ids)

        return self.layer_norm(
            token_vectors + position_vectors
        )