"""Low-level Transformer components implemented with Pytorch tensors."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class TokenEmbedding(Module):
    """Embedding lookup implemented without torch.nn.Embedding."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.weight = Parameter(
            torch.randn(vocab_size, hidden_size)
            * (1.0 / math.sqrt(hidden_size))
        )
    
    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
        
        if torch.any(input_ids < 0):
            raise ValueError("input_ids must be non-negative")
        
        if torch.any(input_ids >= self.vocab_size):
            raise ValueError("input_ids exceed vocabulary size")
        
        return self.weight[input_ids]
    
class Linear(Module):
    """Linear projection implemented without torch.nn.Linear."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if input_size < 1:
            raise ValueError("input_size must be positive")
        
        if output_size < 1:
            raise ValueError("output_size must be positive")
        
        self.input_size = input_size
        self.output_size = output_size

        self.weight = Parameter(
            torch.randn(output_size, input_size)
            * (1.0 / math.sqrt(input_size))
        )

        if bias:
            self.bias = Parameter(torch.zeros(output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.input_size:
            raise ValueError(
                "last input dimension must equal input_size"
            )
        
        outputs = inputs @ self.weight.transpose(-1, -2)

        if self.bias is not None:
            outputs = outputs + self.bias
        
        return outputs

class LayerNorm(Module):
    """Layer normalization implemented without torch.nn.LayerNorm."""

    def __init__(
        self,
        hidden_size: int,
        *,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()

        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        
        self.hidden_size = hidden_size
        self.epsilon = epsilon

        self.weight = Parameter(torch.ones(hidden_size))
        self.bias = Parameter(torch.zeros(hidden_size))
    
    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.hidden_size:
            raise ValueError(
                "last input dimension must equal hidden_size"
            )
        
        mean = inputs.mean(dim=-1, keepdim=True)
        centered = inputs - mean
        variance = (centered * centered).mean(
            dim=-1,
            keepdim=True,
        )

        normalized = centered / torch.sqrt(
            variance + self.epsilon
        )

        return self.weight * normalized + self.bias