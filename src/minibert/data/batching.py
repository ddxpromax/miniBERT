"""Batch variable-length training sequences with padding."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from minibert.data.sequences import PAD_ID


@dataclass
class PaddedBatch:
    """A batch of padded token sequences."""

    input_ids: np.ndarray
    attention_mask: np.ndarray

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]
    
    @property
    def sequence_length(self) -> int:
        return self.input_ids.shape[1]
    
def iter_padded_batches(
    sequences: Iterable[np.ndarray],
    batch_size: int,
) -> Iterator[PaddedBatch]:
    """Group sequences into batches and pad each batch independently."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    
    pending: list[np.ndarray] = []

    for sequence in sequences:
        if sequence.ndim != 1:
            raise ValueError("each sequence must be one-dimensional")
        
        if sequence.dtype != np.dtype("int64"):
            raise ValueError("each sequence must use int64 token IDs")
        
        if len(sequence) == 0:
            raise ValueError("sequences must not be empty")
        
        pending.append(sequence)

        if len(pending) == batch_size:
            yield _make_batch(pending)
            pending = []
        
    if pending:
        yield _make_batch(pending)

def _make_batch(sequences: list[np.ndarray]) -> PaddedBatch:
    max_length = max(len(sequence) for sequence in sequences)

    input_ids = np.full(
        (len(sequences), max_length),
        fill_value=PAD_ID,
        dtype=np.int64,
    )
    attention_mask = np.zeros(
        (len(sequences), max_length),
        dtype=np.int64,
    )

    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = sequence
        attention_mask[row, :length] = 1
    
    return PaddedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )