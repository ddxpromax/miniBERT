"""Streaming document encoding for pretraining."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from minibert.tokenizer.bert import BertTokenizer


def iter_encoded_documents(
    texts: Iterable[str],
    tokenizer: BertTokenizer,
) -> Iterator[list[int]]:
    """Yield token IDs for each non-empty tokenized document.
    
    Document boundaries are preserved. No special tokens are 
    automatically added, and no truncation or padding is applied.
    """
    for text in texts:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        
        if token_ids:
            yield token_ids