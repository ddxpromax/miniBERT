"""Training loops for masked language modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from minibert.data.batching import iter_padded_batches
from minibert.data.masking import apply_mlm_mask
from minibert.data.sequences import iter_training_sequences
from minibert.data.token_storage import TokenizedCorpus
from minibert.model.losses import masked_language_model_loss


@dataclass
class EpochMetrics:
    """Aggregated metrics for one pass over a corpus."""

    loss: float
    batches: int
    masked_positions: int

def _to_device(
    array: np.ndarray,
    device: torch.device,
) -> Tensor:
    """Convert a Numpy array to a long PyTorch tensor."""
    return torch.from_numpy(array).to(device)

def train_epoch(
    model: Module,
    corpus: TokenizedCorpus,
    optimizer: Optimizer,
    *,
    max_length: int,
    batch_size: int,
    vocabulary_size: int,
    rng: np.random.Generator,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Train for one streaming epoch."""
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    
    model.train()

    sequences = iter_training_sequences(
        corpus,
        max_length=max_length,
    )

    batches = iter_padded_batches(
        sequences,
        batch_size=batch_size,
    )

    total_loss = 0.0
    total_masked_positions = 0
    batch_count = 0

    for batch in batches:
        if (
            max_batches is not None
            and batch_count >= max_batches
        ):
            break
        
        masked = apply_mlm_mask(
            batch.input_ids,
            batch.attention_mask,
            vocab_size=vocabulary_size,
            rng=rng,
        )

        input_ids = _to_device(
            masked.input_ids,
            device,
        )
        attention_mask = _to_device(
            batch.attention_mask,
            device,
        )
        labels = _to_device(
            masked.labels,
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            input_ids,
            attention_mask,
        )

        loss = masked_language_model_loss(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        masked_count = int(
            (labels != -100).sum().item()
        )

        total_loss += float(loss.detach()) * masked_count
        total_masked_positions += masked_count
        batch_count += 1

    if total_masked_positions == 0:
        raise ValueError("epoch did not produce any masked positions")
    
    return EpochMetrics(
        loss=total_loss / total_masked_positions,
        batches=batch_count,
        masked_positions=total_masked_positions,
    )

@torch.no_grad()
def evaluate_epoch(
    model: Module,
    corpus: TokenizedCorpus,
    *,
    max_length: int,
    batch_size: int,
    vocabulary_size: int,
    rng: np.random.Generator,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Evaluate one streaming pass without updating parameters."""
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    
    model.eval()

    sequences = iter_training_sequences(
        corpus,
        max_length=max_length,
    )

    batches = iter_padded_batches(
        sequences,
        batch_size=batch_size,
    )

    total_loss = 0.0
    total_masked_positions = 0
    batch_count = 0

    for batch in batches:
        if (
            max_batches is not None
            and batch_count >= max_batches
        ):
            break
        
        masked = apply_mlm_mask(
            batch.input_ids,
            batch.attention_mask,
            vocab_size=vocabulary_size,
            rng=rng,
        )

        input_ids = _to_device(
            masked.input_ids,
            device,
        )
        attention_mask = _to_device(
            batch.attention_mask,
            device,
        )
        labels = _to_device(
            masked.labels,
            device,
        )

        logits = model(
            input_ids,
            attention_mask,
        )

        loss = masked_language_model_loss(
            logits,
            labels,
        )

        masked_count = int(
            (labels != -100).sum().item()
        )

        total_loss += float(loss) * masked_count
        total_masked_positions += masked_count
        batch_count += 1
    
    if total_masked_positions == 0:
        raise ValueError("evaluation did not produce any masked positions")
    
    return EpochMetrics(
        loss=total_loss / total_masked_positions,
        batches=batch_count,
        masked_positions=total_masked_positions,
    )