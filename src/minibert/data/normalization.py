"""Utilities for deterministic corpus text normalization."""

from __future__ import annotations

import re
import unicodedata


def _is_invalid_or_control(character: str) -> bool:
    """Return True for characters that should not enter the training corpus."""
    if ord(character) in (0, 0xFFFD):
        return True
    
    return unicodedata.category(character) in {"Cc", "Cf"}

def normalize_document(text: str) -> str | None:
    """Normalize one raw document without changing its linguistic content.
    
    The returned document has:
    - Unicode compatibility normalization(NFKC);
    - control and formatting characters removed;
    - every kind of whitespace converted to one ASCII space;
    - repeated spaces collapsed;
    - leading and trailing spaces removed.
    
    An input that becomes empty returns None."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    normalized = unicodedata.normalize("NFKC", text)

    characters: list[str] = []
    for character in normalized:
        if character.isspace():
            characters.append(" ")
        elif _is_invalid_or_control(character):
            continue
        else:
            characters.append(character)
    
    result = re.sub(r" +", " ", "".join(characters)).strip()
    return result if result else None