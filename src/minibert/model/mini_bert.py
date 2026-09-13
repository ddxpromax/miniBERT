"""MiniBERT assembled from custom Transformer components."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter

from minibert.model.components import LayerNorm, Linear
from minibert.model.embeddings import BertEmbeddings
from minibert.model.encoder import TransformerEncoderBlock, gelu

class MiniBERT(Module):
    """A small BERT encoder with an MLM prediction head."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_positions: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            max_positions=max_positions,
            hidden_size=hidden_size,
        )

        self.encoder_layers = ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.mlm_transform = Linear(
            hidden_size,
            hidden_size,
        )

        self.mlm_norm = LayerNorm(hidden_size)

        self.mlm_bias = Parameter(
            torch.zeros(vocab_size)
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return MLM logits with shape [batch, sequence, vocab]."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
        
        if attention_mask.dtype != torch.long:
            raise TypeError("attention_mask must use torch.long")
        
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids")
        
        if input_ids.shape[1] > self.max_positions:
            raise ValueError("sequence length exceeds max_positions")

        hidden_states = self.embeddings(input_ids)

        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
            )
        
        hidden_states = self.mlm_transform(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.mlm_norm(hidden_states)

        logits = hidden_states @ (
            self.embeddings.token_embedding.weight.T
        )

        logits = logits + self.mlm_bias

        return logits