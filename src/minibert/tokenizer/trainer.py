"""A deterministic WordPiece vocabulary trainer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from minibert.tokenizer.basic import BasicTokenizer
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS, Vocabulary


def _initial_pieces(word: str) -> tuple[str, ...]:
    """Represent a word as first character plus continuation characters."""
    return tuple(
        character if index == 0 else f"##{character}" 
        for index, character in enumerate(word)
    )

def _merge_pair(
    pieces: tuple[str, ...],
    pair: tuple[str, str],
) -> tuple[str, ...]:
    """Merge every non-overlapping occurrence of one pair in a word."""
    merged: list[str] = []
    index = 0

    while index < len(pieces):
        if (
            index + 1 < len(pieces)
            and (pieces[index], pieces[index + 1]) == pair
        ):
            left, right = pair
            merged.append(left + right.removeprefix("##"))
            index += 2
        else:
            merged.append(pieces[index])
            index += 1

    return tuple(merged)

class WordPieceTrainer:
    """Train a small deterministic WordPiece vocabulary from text documents."""

    def __init__(
        self, 
        vocab_size: int,
        do_lower_case: bool = True,
        min_frequency: int = 2,
    ) -> None:
        if vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError(
                f"vocab_size must be greater than {len(SPECIAL_TOKENS)}"
            )
        
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        
        self.vocab_size = vocab_size
        self.do_lower_case = do_lower_case
        self.min_frequency = min_frequency
    
    def train(self, documents: Iterable[str]) -> Vocabulary:
        """Train and return a WordPiece vocabulary from text documents."""
        basic_tokenizer = BasicTokenizer(
            do_lower_case=self.do_lower_case,
            never_split=set(SPECIAL_TOKENS),
        )
        word_counts: Counter[str] = Counter()

        for document in documents:
            if not isinstance(document, str):
                raise TypeError(
                    "every document must be str, "
                    f"got {type(document).__name__}"
                )
            
            for token in basic_tokenizer.tokenize(document):
                if token not in SPECIAL_TOKENS:
                    word_counts[token] += 1
        
        word_counts = Counter(
            {
                word: count
                for word, count in word_counts.items()
                if count >= self.min_frequency
            }
        )

        if not word_counts:
            raise ValueError("no tokens remain after frequency filtering")
        
        word_pieces = {
            word: _initial_pieces(word)
            for word in word_counts
        }

        initial_tokens = sorted(
            {
                piece
                for pieces in word_pieces.values()
                for piece in pieces
            }
        )
        vocabulary_tokens = [*SPECIAL_TOKENS, *initial_tokens]

        if len(vocabulary_tokens) > self.vocab_size:
            raise ValueError(
                "vocab_size is too small for special tokens "
                "and the initial character vocabulary"
            )
        
        while len(vocabulary_tokens) < self.vocab_size:
            piece_counts: Counter[str] = Counter()
            pair_counts: Counter[tuple[str, str]] = Counter()

            for word, pieces in word_pieces.items():
                frequency = word_counts[word]

                for piece in pieces:
                    piece_counts[piece] += frequency
                
                for index in range(len(pieces) - 1):
                    pair = (pieces[index], pieces[index + 1])
                    pair_counts[pair] += frequency

            candidates: list[tuple[float, int, tuple[str, str]]] = []

            for pair, pair_frequency in pair_counts.items():
                left, right = pair
                score = pair_frequency / (piece_counts[left] * piece_counts[right])
                candidates.append((score, pair_frequency, pair))

            if not candidates: 
                break
            
            _, _, best_pair = min(
                candidates, 
                key=lambda item: (-item[0], -item[1], item[2]),
            )

            new_piece = best_pair[0] + best_pair[1].removeprefix("##")
            vocabulary_tokens.append(new_piece)

            word_pieces = {
                word: _merge_pair(pieces, best_pair)
                for word, pieces in word_pieces.items()
            }

        return Vocabulary(vocabulary_tokens)