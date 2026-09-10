"""Build BERT training sequences from tokenized documents."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from minibert.data.token_storage import TokenizedCorpus
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS

CLS_ID = SPECIAL_TOKENS.index("[CLS]")
SEP_ID = SPECIAL_TOKENS.index("[SEP]")
PAD_ID = SPECIAL_TOKENS.index("[PAD]")


def iter_training_sequences(
    corpus: TokenizedCorpus,
    max_length: int,
) -> Iterator[np.ndarray]:
    """Yield document-local sequences with [CLS] and [SEP].
    
    The returned sequence length is at most ``max_length``.
    No padding and no masking are performed here.
    """
    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    
    content_length = max_length - 2

    for document_index in range(len(corpus)):
        document = corpus[document_index]

        for start in range(0, len(document), content_length):
            content = document[start:start + content_length]

            sequence = np.empty(
                len(content) + 2,
                dtype=np.int64,
            )
            sequence[0] = CLS_ID
            sequence[1:-1] = content
            sequence[-1] = SEP_ID

            yield sequence