"""Transformer encoder blocks implemented from basic components."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module

from minibert.model.attention import MultiHeadSelfAttention
from minibert.model.components import LayerNorm, Linear


def gelu(inputs: Tensor) -> Tensor:
    """Gaussian Error Linear Unit using the tanh approximation."""
    coefficient = math.sqrt(2.0 / math.pi)

    return 0.5 * inputs * (1.0 + torch.tanh(coefficient * (inputs + 0.044715 * inputs.pow(3))))

class FeedForward(Module):
    """The two-layer Transformer feed-forward network."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()

        self.input_projection = Linear(
            hidden_size,
            intermediate_size,
        )
        self.output_projection = Linear(
            intermediate_size,
            hidden_size,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.input_projection(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.output_projection(hidden_states)

        return hidden_states

class TransformerEncoderBlock(Module):
    """One BERT-style Transformer encoder block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.attention_norm = LayerNorm(hidden_size)

        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        self.output_norm = LayerNorm(hidden_size)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
        )

        hidden_states = self.attention_norm(
            hidden_states + attention_output
        )

        hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        feed_forward_output = self.feed_forward(hidden_states)

        hidden_states = self.output_norm(
            hidden_states + feed_forward_output
        )

        hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        return hidden_states