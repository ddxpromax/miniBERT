import json
from pathlib import Path

import numpy as np
import pytest

from minibert.data.token_storage import write_tokenized_corpus
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS, Vocabulary
from minibert.data.token_storage import TokenizedCorpus, write_tokenized_corpus


@pytest.fixture
def vocabulary() -> Vocabulary:
    return Vocabulary([*SPECIAL_TOKENS, "hello", "world"])

def test_storage_preserves_tokens_and_boundaries(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    output_dir = tmp_path / "encoded"

    write_tokenized_corpus(
        iter([[5, 6], [6], [5, 5, 6]]),
        output_dir,
        vocabulary,
    )

    tokens = np.fromfile(output_dir / "tokens.bin", dtype="<u2")
    offsets = np.load(
        output_dir / "offsets.npy",
        allow_pickle=False,
    )
    
    assert tokens.tolist() == [5, 6, 6, 5, 5, 6]
    assert offsets.tolist() == [0, 2, 3, 6]
    assert tokens[offsets[1]:offsets[2]].tolist() == [6]
    assert (output_dir / "tokens.bin").stat().st_size == 12

    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    
    assert metadata["num_documents"] == 3
    assert metadata["num_tokens"] == 6
    assert metadata["vocab_size"] == len(vocabulary)
    assert metadata["token_dtype"] == "<u2"

    saved_vocabulary = Vocabulary.from_file(output_dir / "vocab.txt")
    assert saved_vocabulary.tokens == vocabulary.tokens

def test_storage_skips_empty_documents(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    output_dir = tmp_path / "encoded"

    write_tokenized_corpus(
        [[], [5], [], [6]],
        output_dir,
        vocabulary,
    )

    offsets = np.load(output_dir / "offsets.npy", allow_pickle=False)
    assert offsets.tolist() == [0, 1, 2]

def test_storage_handles_empty_corpus(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    output_dir = tmp_path / "encoded"

    write_tokenized_corpus([], output_dir, vocabulary)

    assert (output_dir / "tokens.bin").stat().st_size == 0

    offsets = np.load(output_dir / "offsets.npy", allow_pickle=False)
    assert offsets.tolist() == [0]

    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["num_documents"] == 0
    assert metadata["num_tokens"] == 0

@pytest.mark.parametrize("identifier", [-1, 7, 65536, 1.5, True])
def test_storage_rejects_invalid_ids(
    tmp_path: Path,
    vocabulary: Vocabulary,
    identifier: object,
) -> None:
    output_dir = tmp_path / "encoded"

    with pytest.raises(ValueError, match="within vocabulary range"):
        write_tokenized_corpus(
            [[identifier]],
            output_dir,
            vocabulary,
        )
    
    assert not (output_dir / "metadata.json").exists()

def test_storage_refuses_existing_directory(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    output_dir = tmp_path / "encoded"
    output_dir.mkdir()
    existing_file = output_dir / "tokens.bin"
    existing_file.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        write_tokenized_corpus([[5]], output_dir, vocabulary)

    assert existing_file.read_bytes() == b"keep"

def test_reader_returns_documents(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    output_dir = tmp_path / "encoded"

    write_tokenized_corpus(
        [[5, 6], [6], [5, 5, 6]],
        output_dir,
        vocabulary,
    )

    corpus = TokenizedCorpus(output_dir)

    assert len(corpus) == 3
    assert corpus.num_tokens == 6
    assert corpus[0].tolist() == [5, 6]
    assert corpus[1].tolist() == [6]
    assert corpus[2].tolist() == [5, 5, 6]
    assert corpus[0].dtype == np.dtype("int64")
    assert corpus.vocabulary.tokens == vocabulary.tokens

def test_reader_returns_independent_copy(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    output_dir = tmp_path / "encoded"
    write_tokenized_corpus([[5, 6]], output_dir, vocabulary)

    corpus = TokenizedCorpus(output_dir)
    document = corpus[0]

    document[0] = vocabulary.token_id("[MASK]")

    assert corpus[0].tolist() == [5, 6]

    stored = np.fromfile(output_dir / "tokens.bin", dtype="<u2")
    assert stored.tolist() == [5, 6]

def test_reader_handles_empty_corpus(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    output_dir = tmp_path / "encoded"
    write_tokenized_corpus([], output_dir, vocabulary)

    corpus = TokenizedCorpus(output_dir)

    assert len(corpus) == 0
    assert corpus.num_tokens == 0

    with pytest.raises(IndexError):
        corpus[0]

@pytest.mark.parametrize("index", [-1, 1])
def test_reader_rejects_out_of_range_index(
    tmp_path: Path,
    vocabulary: Vocabulary,
    index: int,
) -> None:
    output_dir = tmp_path / "encoded"
    write_tokenized_corpus([[5]], output_dir, vocabulary)

    corpus = TokenizedCorpus(output_dir)

    with pytest.raises(IndexError, match="out of range"):
        corpus[index]


def test_reader_rejects_truncated_token_file(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    output_dir = tmp_path / "encoded"
    write_tokenized_corpus([[5, 6]], output_dir, vocabulary)

    (output_dir / "tokens.bin").write_bytes(b"\x05\x00")

    with pytest.raises(ValueError, match="token file size"):
        TokenizedCorpus(output_dir)

def test_reader_rejects_invalid_boundaries(
    tmp_path: Path,
    vocabulary: Vocabulary,
) -> None:
    output_dir = tmp_path / "encoded"
    write_tokenized_corpus([[5], [6]], output_dir, vocabulary)

    np.save(
        output_dir / "offsets.npy",
        np.asarray([0, 3, 2], dtype="<u8"),
        allow_pickle=False,
    )

    with pytest.raises(ValueError, match="document boundaries"):
        TokenizedCorpus(output_dir)