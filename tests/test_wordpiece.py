import pytest

from minibert.tokenizer.wordpiece import WordPieceTokenizer


@pytest.fixture
def tokenizer() -> WordPieceTokenizer:
    vocabulary = {
        "[UNK]",
        "un",
        "##want",
        "##ed",
        "want",
        "##ed",
        "play",
        "##ing",
        "hello",
        "world",
        ",",
    }
    return WordPieceTokenizer(vocabulary)

def test_wordpiece_uses_greedy_longest_match(tokenizer: WordPieceTokenizer) -> None:
    assert tokenizer.tokenize("unwanted") == ["un", "##want", "##ed"]

def test_wordpiece_splits_word_into_root_and_suffix(
    tokenizer: WordPieceTokenizer,
) -> None:
    assert tokenizer.tokenize("playing") == ["play", "##ing"]

def test_wordpiece_keeps_complete_vocabulary_word(
    tokenizer: WordPieceTokenizer,
) -> None:
    assert tokenizer.tokenize("hello") == ["hello"]

def test_wordpiece_returns_unknown_when_full_word_cannot_be_split(
    tokenizer: WordPieceTokenizer,
) -> None:
    assert tokenizer.tokenize("unknown") == ["[UNK]"]

def test_wordpiece_returns_unknown_for_excessively_long_token() -> None:
    tokenizer = WordPieceTokenizer(
        vocabulary={"[UNK]", "a", "##a"},
        max_input_characters_per_word=5,
    )

    assert tokenizer.tokenize("aaaaaa") == ["[UNK]"]

def test_wordpiece_validates_constructor_and_input() -> None:
    with pytest.raises(ValueError, match="must exist in the vocabulary"):
        WordPieceTokenizer({"hello"})
    
    with pytest.raises(ValueError, match="must be at least 1"):
        WordPieceTokenizer({"[UNK]"}, max_input_characters_per_word=0)
    
    tokenizer = WordPieceTokenizer({"[UNK]", "hello"})
    with pytest.raises(TypeError, match="token must be str"):
        tokenizer.tokenize(None)