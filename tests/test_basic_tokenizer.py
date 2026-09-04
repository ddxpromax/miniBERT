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