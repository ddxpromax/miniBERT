from collections.abc import Iterator

import pytest

from minibert.data.encoding import iter_encoded_documents
from minibert.tokenizer.bert import BertTokenizer
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS, Vocabulary


@pytest.fixture
def tokenizer() -> BertTokenizer:
    vocabulary = Vocabulary(
        [
            *SPECIAL_TOKENS,
            "hello",
            "world",
            "play",
            "##ing",
        ]
    )
    return BertTokenizer(vocabulary)

def test_encoding_preserves_document_boundaries(
    tokenizer: BertTokenizer,
) -> None:
    result = list(
        iter_encoded_documents(
            ["Hello world", "Playing"],
            tokenizer,
        )
    )
    
    assert result == [[5, 6], [7, 8]]

@pytest.mark.parametrize("text", ["", " ", "\t\n"])
def test_encoding_skips_empty_documents(
    tokenizer: BertTokenizer,
    text: str,
) -> None:
    result = list(
        iter_encoded_documents(
            ["Hello", text, "World"],
            tokenizer,
        )
    )

    assert result == [[5], [6]]

def test_encoding_preserves_unknown_tokens(
    tokenizer: BertTokenizer,
) -> None:
    result = list(iter_encoded_documents(["missing"], tokenizer))

    assert result == [[tokenizer.vocabulary.token_id("[UNK]")]]

def test_encoding_does_not_truncate_long_documents(
    tokenizer: BertTokenizer,
) -> None:
    result = list(iter_encoded_documents(["hello " * 600], tokenizer))

    assert result == [[5] * 600]

def test_encoding_accepts_empty_input(
    tokenizer: BertTokenizer,
) -> None:
    assert list(iter_encoded_documents([], tokenizer)) == []

def test_encoding_consumes_documents_lazily(
    tokenizer: BertTokenizer,
) -> None:
    consumed: list[str] = []

    def text() -> Iterator[str]:
        for text in ["Hello", "World"]:
            consumed.append(text)
            yield text
    
    documents = iter_encoded_documents(text(), tokenizer)

    assert consumed == []
    assert next(documents) == [5]
    assert consumed == ["Hello"]

    assert next(documents) == [6]
    assert consumed == ["Hello", "World"]

    with pytest.raises(StopIteration):
        next(documents)