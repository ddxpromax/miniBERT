import pytest

from minibert.tokenizer.basic import BasicTokenizer


def test_uncased_tokenization_matches_bert_style() -> None:
    tokenizer = BasicTokenizer()

    assert tokenizer.tokenize("UNwantéd,running") == [
        "unwanted",
        ",",
        "running",
    ]

def test_tokenizer_split_ascii_and_unicode_punctuation() -> None:
    tokenizer = BasicTokenizer()

    assert tokenizer.tokenize("Hello... world! Isn't this (fun)?") == [
        "hello", 
        ".",
        ".",
        ".",
        "world",
        "!",
        "isn",
        "'",
        "t",
        "this",
        "(",
        "fun",
        ")",
        "?",
    ]

def test_tokenizer_normalizes_whitespace_and_removes_controls() -> None:
    tokenizer = BasicTokenizer()

    assert tokenizer.tokenize("  hello\tworld\u0000 \u200btest\n") == [
        "hello", 
        "world",
        "test",
    ]

def test_tokenizer_preservers_never_split_tokens() -> None:
    tokenizer = BasicTokenizer(never_split={"[CLS]", "[SEP]"})

    assert tokenizer.tokenize("[CLS] Hello [SEP]") == ["[CLS]", "hello", "[SEP]"]

def test_tokenizer_can_keep_case_and_accents() -> None:
    tokenizer = BasicTokenizer(do_lower_case=False)

    assert tokenizer.tokenize("Café's Here") == ["Café", "'", "s", "Here"]

def test_tokenizer_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="text must be str"):
        BasicTokenizer().tokenize(None)

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026", ["2", "0", "2", "6"]),
        ("abc2026def", ["abc", "2", "0", "2", "6", "def"]),
        ("B52", ["b", "5", "2"]),
        ("-3.14%", ["-", "3", ".", "1", "4", "%"]),
        ("007", ["0", "0", "7"]),
        ("abc１２٣def", ["abc", "1", "2", "3", "def"]),
    ],
)
def test_tokenizer_isolates_decimal_digits(
    text: str,
    expected: list[str],
) -> None:
    assert BasicTokenizer().tokenize(text) == expected

def test_digit_splitting_preserves_never_split_tokens() -> None:
    tokenizer = BasicTokenizer(never_split={"[SPECIAL2]"})

    assert tokenizer.tokenize("[SPECIAL2] B52") == [
        "[SPECIAL2]",
        "b",
        "5",
        "2",
    ]