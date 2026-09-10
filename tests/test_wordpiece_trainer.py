import pytest

from minibert.tokenizer.bert import BertTokenizer
from minibert.tokenizer.trainer import WordPieceTrainer
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS, Vocabulary


def test_trainer_creates_requested_size_and_fixed_special_tokens() -> None:
    trainer = WordPieceTrainer(vocab_size=30, min_frequency=1)

    vocabulary = trainer.train(
        [
            "Playing played playing.",
            "Players play.",
        ]
    )

    assert len(vocabulary) == 30
    assert vocabulary.tokens[:5] == SPECIAL_TOKENS

def test_trainer_id_deterministic() -> None:
    documents = [
        "Playing played playing.",
        "Players play.",
    ]

    first = WordPieceTrainer(vocab_size=30, min_frequency=1).train(documents)
    second = WordPieceTrainer(vocab_size=30, min_frequency=1).train(documents)

    assert first.tokens == second.tokens

def test_trained_vocabulary_can_encode_seen_text() -> None:
    vocabulary = WordPieceTrainer(vocab_size=30, min_frequency=1).train(
        [
            "Playing played playing.",
            "Players play.",
        ]
    )
    tokenizer = BertTokenizer(vocabulary)

    tokens = tokenizer.tokenize("Playing")

    assert tokens != ["[UNK]"]
    assert "".join(piece.removeprefix("##") for piece in tokens) == "playing"

def test_trainer_filters_infrequent_words() -> None:
    trainer = WordPieceTrainer(vocab_size=22, min_frequency=2)

    vocabulary = trainer.train(
        [
            "common common rare",
            "common",
        ]
    )

    assert "r" not in vocabulary.tokens
    assert "##r" not in vocabulary.tokens

def test_trainer_rejects_invalid_configuration_and_documents() -> None:
    with pytest.raises(ValueError, match="vocab_size must be greater"):
        WordPieceTrainer(vocab_size=5)
    
    with pytest.raises(ValueError, match="min_frequency must be at least 1"):
        WordPieceTrainer(vocab_size=10, min_frequency=0)

    trainer = WordPieceTrainer(vocab_size=10, min_frequency=1)

    with pytest.raises(TypeError, match="every document must be str"):
        trainer.train(["valid", None])

    with pytest.raises(ValueError, match="no tokens remain"):
        WordPieceTrainer(vocab_size=10, min_frequency=2).train(["once"])

@pytest.mark.parametrize(
    ("vocab_size", "expected_merges"),
    [
        (19, []),
        (20, ["ac"]),
        (21, ["ac", "ab"]),
        (22, ["ac", "ab", "db"]),
        (30, ["ac", "ab", "db"]),
    ],
)
def test_trainer_produces_expected_merge_sequence(
    vocab_size: int,
    expected_merges: list[str],
) -> None:
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
    )

    vocabulary = trainer.train(["ab ac db"])

    assert vocabulary.tokens == (
        *SPECIAL_TOKENS,
        "##b",
        "##c",
        *"0123456789",
        "a",
        "d",
        *expected_merges,
    )

def test_trainer_id_deterministic_across_document_order() -> None:
    documents = [
        "Playing played playing.",
        "Players play.",
    ]
    trainer = WordPieceTrainer(vocab_size=30, min_frequency=1)

    first = trainer.train(documents)
    second = trainer.train(reversed(documents))

    assert first.tokens == second.tokens

def test_trainer_and_encoder_respect_digit_boundaries() -> None:
    trainer = WordPieceTrainer(vocab_size=64, min_frequency=2)

    vocabulary = trainer.train(["abc2026def abc2026def"])
    tokenizer = BertTokenizer(vocabulary)

    assert tokenizer.tokenize("abc2026def") == [
        "abc", "2", "0", "2", "6", "def",
    ]

    assert "abcdef" not in vocabulary.tokens

    digit_tokens = {
        token
        for token in vocabulary.tokens
        if any(character.isdecimal() for character in token)
    }
    assert digit_tokens == set("0123456789")

def test_trainer_excludes_non_latin_text_and_preserves_boundaries() -> None:
    trainer = WordPieceTrainer(vocab_size=64, min_frequency=2)

    vocabulary = trainer.train(
        ["ab中文cd ab中文cd 東京 Русский"]
    )
    tokenizer = BertTokenizer(vocabulary)

    assert all(token.isascii() for token in vocabulary.tokens)
    assert "ab" in vocabulary.tokens
    assert "cd" in vocabulary.tokens
    
    assert "abcd" not in vocabulary.tokens

    assert tokenizer.tokenize("ab中文cd") == [
        "ab", "[UNK]", "cd",
    ]
    assert tokenizer.encode("ab中文cd") == [
        vocabulary.token_id("[CLS]"),
        vocabulary.token_id("ab"),
        vocabulary.token_id("[UNK]"),
        vocabulary.token_id("cd"),
        vocabulary.token_id("[SEP]"),
    ]

def test_trainer_rejects_corpus_with_only_unsupported_letters() -> None:
    trainer = WordPieceTrainer(vocab_size=64, min_frequency=1)

    with pytest.raises(ValueError, match="no tokens remain"):
        trainer.train(["中文 東京 Русский"])