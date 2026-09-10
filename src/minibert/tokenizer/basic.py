"""Basic tokenization used before WorkPiece segmentation."""

from __future__ import annotations

import unicodedata
import regex

ASCII_DIGITS = "0123456789"
NON_LATIN_RUN = regex.compile(
    r"(?:[\p{Letter}--\p{Script=Latin}]\p{Mark}*)+",
    flags=regex.VERSION1,
)


def _is_whitespace(character: str) -> bool:
    """Return whether a character is treated as whitespace."""
    return character in {" ", "\t", "\n", "\r"} or unicodedata.category(character) == "Zs"

def _is_control(character: str) -> bool:
    """Return whether a character is a non-whitespace control character."""
    if character in {"\t", "\n", "\r"}:
        return False
    return unicodedata.category(character).startswith("C")

def _is_punctuation(character: str) -> bool:
    """Return whether a character is punctuation."""
    codepoint = ord(character)
    if (
        33 <= codepoint <= 47
        or 58 <= codepoint <= 64
        or 91 <= codepoint <= 96
        or 123 <= codepoint <= 126
    ):
        return True
    
    return unicodedata.category(character).startswith("P")

def _strip_accents(text: str) -> str:
    """Remove Unicode accent marks from a string."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )

def _clean_text(text: str) -> str:
    """Remove invalid characters and normalize whitespace to ASCII spaces."""
    output: list[str] = []

    for character in text:
        if ord(character) in {0, 0xFFFD} or _is_control(character):
            continue
        
        if _is_whitespace(character):
            output.append(" ")
        else: 
            output.append(character)
    
    return "".join(output)

def _split_on_punctuation(text: str) -> list[str]:
    """Split text so that every punctuation character becomes its own token."""
    tokens: list[list[str]] = []
    current_token: list[str] = []

    for character in text:
        if _is_punctuation(character):
            if current_token:
                tokens.append(current_token)
                current_token = []
            tokens.append([character])
        else:
            current_token.append(character)
    
    if current_token:
        tokens.append(current_token)
    
    return ["".join(token) for token in tokens]

def _split_on_digits(text: str) -> list[str]:
    """Isolate decimal digits and normalize them to ASCII."""
    tokens: list[str] = []
    current: list[str] = []

    for character in text:
        if character.isdecimal():
            if current:
                tokens.append("".join(current))
                current = []

            digit = unicodedata.decimal(character)
            tokens.append(str(digit))
        else:
            current.append(character)

    if current:
        tokens.append("".join(current))
    
    return tokens

def _replace_non_latin_runs(text: str) -> list[str]:
    """Replace non-Latin letter runs with seperate unknown tokens."""
    tokens: list[str] = []
    position = 0

    for match in NON_LATIN_RUN.finditer(text):
        if match.start() > position:
            tokens.append(text[position:match.start()])

        tokens.append("[UNK]")
        position = match.end()

    if position < len(text):
        tokens.append(text[position:])
    return tokens

class BasicTokenizer:
    """Tokenize text using the uncased BERT basic-tokenization rules."""

    def __init__(
        self, 
        do_lower_case: bool = True,
        never_split: set[str] | None = None,
    ) -> None:
        self.do_lower_case = do_lower_case
        self.never_split = never_split or set()
    
    def tokenize(self, text: str) -> list[str]:
        """Convert one string into basic tokens."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        
        cleaned = _clean_text(text)
        whitespace_tokens = cleaned.strip().split()

        output_tokens: list[str] = []
        for token in whitespace_tokens:
            if token in self.never_split:
                output_tokens.append(token)
                continue
            
            if self.do_lower_case:
                token = _strip_accents(token.lower())
            
            for punctuation_token in _split_on_punctuation(token):
                for digit_token in _split_on_digits(punctuation_token):
                    output_tokens.extend(_replace_non_latin_runs(digit_token))
        
        return output_tokens