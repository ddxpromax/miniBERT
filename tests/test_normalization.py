import pytest

from minibert.data.normalization import normalize_document

def test_normalize_document_preserves_meaningful_text() -> None:
    text = "  Hello,\tworld!\nThis is a test."

    assert normalize_document(text) == "Hello, world! This is a test."

def test_normalize_document_applies_unicode_nfkc() -> None:
    text = "ＡＢＣ　１２３"

    assert normalize_document(text) == "ABC 123"

def test_normalize_document_removes_control_characters() -> None:
    text = "hello\x00 world\u200b!"

    assert normalize_document(text) == "hello world!"

def test_normalize_document_keeps_case_accents_and_punctuation() -> None:
    text = "Café -- Don't LOWERCASE me."

    assert normalize_document(text) == "Café -- Don't LOWERCASE me."

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", None),
        (" \t\n\r", None),
        ("\x00\u200b", None),
    ],
)

def test_normalize_document_returns_none_for_empty_result(
    text: str,
    expected: None,
) -> None:
    assert normalize_document(text) == expected

def test_normalize_document_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="text must be str"):
        normalize_document(None)    #type: ignore[arg-type]