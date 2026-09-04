"""Greedy longest-match-first WordPiece tokenization."""

from __future__ import annotations


class WordPieceTokenizer:
    """Split one basic token into WordPiece subword tokens."""

    def __init__(
        self, 
        vocabulary: set[str],
        unknown_token: str = "[UNK]",
        max_input_characters_per_word: int = 100,
    ) -> None:
        if unknown_token not in vocabulary:
            raise ValueError(
                f"unknown_token {unknown_token!r} must exist in the vocabulary"
            )
        
        if max_input_characters_per_word < 1:
            raise ValueError("max_input_characters_per_word must be at least 1")

        self.vocabulary = vocabulary
        self.unknown_token = unknown_token
        self.max_input_characters_per_word = max_input_characters_per_word

    def tokenize(self, token: str) -> list[str]:
        """Split one token using greedy longest-match-first WordPiece."""
        if not isinstance(token, str):
            raise TypeError(f"token must be str, got {type(token).__name__}")
        
        if len(token) > self.max_input_characters_per_word:
            return [self.unknown_token]
        
        start = 0
        subwords: list[str] = []

        while start < len(token):
            end = len(token)
            current_subword: str | None = None

            while start < end:
                substring = token[start:end]
                if start > 0:
                    substring = f"##{substring}"
                
                if substring in self.vocabulary:
                    current_subword = substring
                    break

                end -= 1
            
            if current_subword is None:
                return [self.unknown_token]
            
            subwords.append(current_subword)
            start = end
        
        return subwords