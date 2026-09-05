import pytest

from pathlib import Path

from minibert.tokenizer.bert import BertTokenizer
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS, Vocabulary


@pytest.fixture
def vocabulary() -> Vocabulary:
    return Vocabulary(
        [
            *SPECIAL_TOKENS,
            "hello",
            "world",
            "play",
            "##ing",
            ",",
            "!",
        ]
    )

@pytest.fixture
def tokenizer(vocabulary: Vocabulary) -> BertTokenizer:
    return BertTokenizer(vocabulary)

def test_special_tokens_have_fixed_ids(vocabulary: Vocabulary) -> None:
    assert vocabulary.token_id("[PAD]") == 0
    assert vocabulary.token_id("[UNK]") == 1
    assert vocabulary.token_id("[CLS]") == 2
    assert vocabulary.token_id("[SEP]") == 3
    assert vocabulary.token_id("[MASK]") == 4

def test_vocabulary_uses_unknown_id_for_unknown_token(
    vocabulary: Vocabulary,
) -> None:
    assert vocabulary.token_id("missing") == vocabulary.token_id("[UNK]")

def test_vocabulary_converts_id_to_token(vocabulary: Vocabulary) -> None:
    assert vocabulary.token_for_id(5) == "hello"

    with pytest.raises(ValueError, match="identifier must be between"):
        vocabulary.token_for_id(999)

def test_vocabulary_rejects_duplicate_or_invalid_special_tokens() -> None:
    with pytest.raises(ValueError, match="tokens must be unique"):
        Vocabulary([*SPECIAL_TOKENS, "hello", "hello"])

    with pytest.raises(ValueError, match="first vocabulary tokens"):
        Vocabulary(["hello", *SPECIAL_TOKENS])

def test_bert_tokenizer_combines_basic_and_wordpiece(
    tokenizer: BertTokenizer,
) -> None:
    assert tokenizer.tokenize("Hello, playing!") == [
        "hello", 
        ",",
        "play",
        "##ing",
        "!",
    ]

def test_bert_tokenizer_encodes_with_special_tokens(
    tokenizer: BertTokenizer,
) -> None:
    assert tokenizer.encode("Hello, playing!") == [
        2,
        5,
        9,
        7,
        8,
        10,
        3,
    ]

def test_bert_tokenizer_encodes_without_special_tokens(
    tokenizer: BertTokenizer,
) -> None:
    assert tokenizer.encode("Hello", add_special_tokens=False) == [5]

def test_bert_tokenizer_preserves_special_tokens_in_input(
    tokenizer: BertTokenizer,
) -> None:
    assert tokenizer.tokenize("[CLS] Hello [SEP]") == [
        "[CLS]",
        "hello",
        "[SEP]",
    ]

def test_vocabulary_save_and_load_round_trip(
    vocabulary: Vocabulary,
    tmp_path: Path,
) -> None:
    vocabulary_path = tmp_path / "vocab.txt"

    vocabulary.save(vocabulary_path)
    loaded = Vocabulary.from_file(vocabulary_path)

    assert loaded.tokens == vocabulary.tokens
    assert loaded.token_to_id == vocabulary.token_to_id

def test_vocabulary_file_has_one_token_per_line(
    vocabulary: Vocabulary,
    tmp_path: Path,
) -> None:
    vocabulary_path = tmp_path / "vocab.txt"

    vocabulary.save(vocabulary_path)

    assert vocabulary_path.read_text(encoding="utf-8").splitlines() == list(
        vocabulary.tokens
    )