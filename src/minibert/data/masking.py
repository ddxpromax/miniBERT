"""Dynamic masked-language-model generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minibert.tokenizer.vocabulary import SPECIAL_TOKENS


PAD_ID = SPECIAL_TOKENS.index("[PAD]")
CLS_ID = SPECIAL_TOKENS.index("[CLS]")
SEP_ID = SPECIAL_TOKENS.index("[SEP]")
MASK_ID = SPECIAL_TOKENS.index("[MASK]")

IGNORE_INDEX = -100


@dataclass
class MaskedBatch:
    """Inputs and labels for one MLM batch."""

    input_ids: np.ndarray
    labels: np.ndarray

def apply_mlm_mask(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    *,
    vocab_size: int,
    rng: np.random.Generator,
    mask_probability: float = 0.15,
) -> MaskedBatch:
    """Craete dynamically masked inputs and MLM labels.
    
    ``input_ids`` and ``attention_mask`` are not modified in place.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be two-dimensional")
    
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    
    if input_ids.dtype != np.dtype("int64"):
        raise ValueError("input_ids must use int64 token IDs")
    
    if attention_mask.dtype != np.dtype("int64"):
        raise ValueError("attention_mask must use int64 values")
    
    if vocab_size <= MASK_ID + 1:
        raise ValueError("vocab_size is too small")

    if not 0.0 < mask_probability <= 1.0:
        raise ValueError("mask_probability must be greater than 0 and at most 1")
    
    masked_inputs = input_ids.copy()
    labels = np.full_like(
        input_ids,
        fill_value=IGNORE_INDEX,
    )

    for row in range(input_ids.shape[0]):
        candidates = np.flatnonzero(
            (attention_mask[row] == 1)
            & (input_ids[row] != PAD_ID)
            & (input_ids[row] != CLS_ID)
            & (input_ids[row] != SEP_ID)
        )
        
        if len(candidates) == 0:
            continue
            
        number_to_mask = max(
            1,
            int(round(len(candidates) * mask_probability)),
        )
        number_to_mask = min(number_to_mask, len(candidates))

        selected = rng.choice(
            candidates,
            size=number_to_mask,
            replace=False,
        )

        labels[row, selected] = input_ids[row, selected]

        decisions = rng.random(number_to_mask)

        mask_positions = selected[decisions < 0.8]
        random_positions = selected[
            (decisions >= 0.8) & (decisions < 0.9)
        ]

        masked_inputs[row, mask_positions] = MASK_ID

        if len(random_positions) > 0:
            masked_inputs[row, random_positions] = rng.integers(
                low=MASK_ID + 1,
                high=vocab_size,
                size=len(random_positions),
                dtype=np.int64,
            )

    return MaskedBatch(
        input_ids=masked_inputs,
        labels=labels,
    )