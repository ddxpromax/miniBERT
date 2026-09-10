from pathlib import Path

import numpy as np
import pytest

from minibert.data.sequences import CLS_ID, SEP_ID, iter_training_sequences
from minibert.data.token_storage import TokenizedCorpus, write_tokenized_corpus
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS, Vocabulary

@pytest.fixture
def vocabulary() -> Vocabulary:
    return Vocabulary(
        [
            *SPECIAL_TOKENS,
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
        ]
    )

def build_corpus(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> TokenizedCorpus:
    output_dir = tmp_path / "encoded"

    write_tokenized_corpus(
        [
            [5, 6, 7, 8],
            [9, 10],
        ],
        output_dir,
        vocabulary,
    )

    return TokenizedCorpus(output_dir)

def test_sequences_add_special_tokens_without_crossing_documents(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    corpus = build_corpus(tmp_path, vocabulary)

    sequences = list(
        iter_training_sequences(
            corpus,
            max_length=6,
        )
    )

    assert [sequence.tolist() for sequence in sequences] == [
        [CLS_ID, 5, 6, 7, 8, SEP_ID],
        [CLS_ID, 9, 10, SEP_ID],
    ]

def test_long_document_is_split_into_multiple_sequences(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    output_dir = tmp_path / "encoded"

    write_tokenized_corpus(
        [[5, 6, 7, 8, 9, 10, 11]],
        output_dir,
        vocabulary,
    )

    corpus = TokenizedCorpus(output_dir)

    sequences = list(
        iter_training_sequences(
            corpus,
            max_length=5,
        )
    )

    assert [sequence.tolist() for sequence in sequences] == [
        [CLS_ID, 5, 6, 7, SEP_ID],
        [CLS_ID, 8, 9, 10, SEP_ID],
        [CLS_ID, 11, SEP_ID],
    ]

def test_sequences_are_int64_and_not_padded(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    corpus = build_corpus(tmp_path, vocabulary)

    sequence = next(iter_training_sequences(corpus, max_length=8))

    assert sequence.dtype == np.dtype("int64")
    assert len(sequence) == 6
    assert 0 not in sequence

@pytest.mark.parametrize("max_length", [0, 1, 2])
def test_sequence_length_must_leave_room_for_special_tokens(
    tmp_path: Path,
    vocabulary: Vocabulary,
    max_length: int,
) -> None:
    corpus = build_corpus(tmp_path, vocabulary)

    with pytest.raises(ValueError, match="at least 3"):
        list(iter_training_sequences(corpus, max_length))

def test_empty_corpus_yields_no_sequences(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    output_dir = tmp_path / "encoded"

    write_tokenized_corpus([], output_dir, vocabulary)
    corpus = TokenizedCorpus(output_dir)

    assert list(iter_training_sequences(corpus, max_length=128)) == []