"""BERT-compatible composition of basic and WordPiece tokenization."""

from __future__ import annotations

from minibert.tokenizer.basic import BasicTokenizer
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS, Vocabulary
from minibert.tokenizer.wordpiece import WordPieceTokenizer


class BertTokenizer:
    """Tokenize uncased text and convert tokens to BERT vocabulary IDs."""

    def __init__(
        self, 
        vocabulary: Vocabulary,
        do_lower_case: bool = True,
    ) -> None:
        self.vocabulary = vocabulary
        self.basic_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case,
            never_split=set(SPECIAL_TOKENS),
        )
        self.wordpiece_tokenizer = WordPieceTokenizer(
            vocabulary=set(vocabulary.tokens),
            unknown_token="[UNK]",
        )
    
    def tokenize(self, text: str) -> list[str]:
        """Convert one raw string into WordPiece tokens."""
        output_tokens: list[str] = []

        for basic_token in self.basic_tokenizer.tokenize(text):
            if basic_token in SPECIAL_TOKENS:
                output_tokens.append(basic_token)
            else:
                output_tokens.extend(self.wordpiece_tokenizer.tokenize(basic_token))

        return output_tokens
    
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert vocabulary tokens into their integer IDs."""
        return [self.vocabulary.token_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, identifiers: list[int]) -> list[str]:
        """Convert integer IDs back into vocabulary tokens."""
        return [
            self.vocabulary.token_for_id(identifier)
            for identifier in identifiers
        ]
    
    def encode(
        self, 
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Tokenize text and return vocabulary IDs."""
        tokens = self.tokenize(text)

        if add_special_tokens:
            tokens = ["[CLS]", *tokens, "[SEP]"]
        
        return self.convert_tokens_to_ids(tokens)