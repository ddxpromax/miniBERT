"""Multi-head self-attention implemented with basic Pytorch operations."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module

from minibert.model.components import Linear


class MultiHeadSelfAttention(Module):
    """BERT-style multi-head self-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        
        if num_heads < 1:
            raise ValueError("num_heads must be positive")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query_projection = Linear(
            hidden_size,
            hidden_size,
        )
        self.key_projection = Linear(
            hidden_size,
            hidden_size,
        )
        self.value_projection = Linear(
            hidden_size,
            hidden_size,
        )
        self.output_projection = Linear(
            hidden_size,
            hidden_size,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return attended hidden states.
        
        hidden_states:
            [batch_size, sequence_length, hidden_size]
        
        attention_mask:
            [batch_size, sequence_length]
            1 means real token, 0 means padding.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        
        batch_size, sequence_length, hidden_size = (
            hidden_states.shape
        )
        
        if hidden_size != self.hidden_size:
            raise ValueError("hidden_states last dimension must equal hidden_size")
        
        if attention_mask.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError("attention_mask shape must match hidden_states")
        
        if attention_mask.dtype != torch.long:
            raise TypeError("attention_mask must use torch.long")
        
        queries = self.query_projection(hidden_states)
        keys = self.key_projection(hidden_states)
        values = self.value_projection(hidden_states)

        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        scores = torch.matmul(
            queries,
            keys.transpose(-1, -2),
        )

        scores = scores / math.sqrt(self.head_size)

        key_mask = attention_mask[:, None, None, :].bool()

        scores = scores.masked_fill(
            ~key_mask,
            torch.finfo(scores.dtype).min,
        )

        probabilities = torch.softmax(
            scores,
            dim=-1,
        )

        attended = torch.matmul(
            probabilities,
            values,
        )

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

        output = self.output_projection(attended)

        output = output * attention_mask.unsqueeze(-1)

        return output

    def _split_heads(self, inputs: Tensor) -> Tensor:
        """[B, L, H] -> [B, heads, L, head_size]."""
        batch_size, sequence_length, _ = inputs.shape

        inputs = inputs.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size,
        )

        return inputs.transpose(1, 2)