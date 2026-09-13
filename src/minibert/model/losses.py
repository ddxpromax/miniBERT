"""Loss functions implemented with basic PyTorch operations."""

from __future__ import annotations

import torch
from torch import Tensor

IGNORE_INDEX = -100


def masked_language_model_loss(
    logits: Tensor,
    labels: Tensor,
) -> Tensor:
    """Compute MLM cross-entropy only at selected positions.
    
    logits:
        [batch_size, sequence_length, vocab_size]
        
    labels:
        [batch_size, sequence_length]
        
    Labels equal to ``IGNORE_INDEX`` are ignored.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")
    
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    
    if logits.shape[:2] != labels.shape:
        raise ValueError("labels shape must match logits batch and sequence dimensions")
    
    if labels.dtype != torch.long:
        raise TypeError("labels must use torch.long")
    
    valid_positions = labels != IGNORE_INDEX

    if not torch.any(valid_positions):
        raise ValueError("at least one label must be different from IGNORE_INDEX")
    
    selected_logits = logits[valid_positions]
    selected_labels = labels[valid_positions]

    vocab_size = logits.shape[-1]

    if torch.any(selected_labels < 0):
        raise ValueError("valid labels must be non-negative")
    
    if torch.any(selected_labels >= vocab_size):
        raise ValueError("valid labels exceed vocabulary size")
    
    log_normalizer = torch.logsumexp(
        selected_logits,
        dim=-1,
    )

    correct_logits = selected_logits[
        torch.arange(
            selected_logits.shape[0],
            device=selected_logits.device,
        ),
        selected_labels,
    ]

    negative_log_likelihood = (
        log_normalizer - correct_logits
    )

    return negative_log_likelihood.mean()