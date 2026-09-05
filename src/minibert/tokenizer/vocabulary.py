"""Vocabulary storage and token-ID conversion."""

from __future__ import annotations

from pathlib import Path

SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")

class Vocabulary:
    """An immutable ordered WordPiece vocabulary."""

    def __init__(self, tokens: list[str]) -> None:
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulary tokens must be unique")

        if tokens[: len(SPECIAL_TOKENS)] != list(SPECIAL_TOKENS):
            raise ValueError(
                "the first vocabulary tokens must be "
                f"{list(SPECIAL_TOKENS)!r}"
            )
        self.tokens = tuple(tokens)
        self.token_to_id = {
            token: identifier
            for identifier, token in enumerate(self.tokens)
        }
    
    def __len__(self) -> int:
        return len(self.tokens)
    
    def contains(self, tokens: str) -> bool:
        """Return whether a token exist in this vocabulary."""
        return token in self.token_to_id

    def token_id(self, token: str) -> int:
        """Return the ID for a token, falling back to [UNK]."""
        return self.token_to_id.get(token, self.token_to_id["[UNK]"])
    
    def token_for_id(self, identifier: int) -> str:
        """Return the token represented by an integer ID."""
        if not isinstance(identifier, int):
            raise TypeError(
                f"identifier must be int, got {type(identifier).__name__}"
            )
        
        if identifier < 0 or identifier >= len(self.tokens):
            raise ValueError(
                f"identifier must be between 0 and {len(self.tokens) - 1}"
            )
        
        return self.tokens[identifier]
    
    def save(self, path: Path) -> None:
        """Save one vocabulary token per UTF-8 line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.tokens) + "\n", encoding="utf-8")

    @classmethod
    def from_file(cls, path: Path) -> "Vocabulary":
        """Load a UTF-8 vocabulary with one token per line."""
        tokens = path.read_text(encoding="utf-8").splitlines()
        return cls(tokens)