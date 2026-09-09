"""A deterministic WordPiece vocabulary trainer."""

from __future__ import annotations

from tqdm import tqdm

from collections import Counter
from collections.abc import Iterable

from minibert.tokenizer.basic import BasicTokenizer, ASCII_DIGITS
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS, Vocabulary
from minibert.tokenizer.pair_heap import PairHeap
from minibert.tokenizer.training_state import IncrementalWordPieceState


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
    
    def train(self, documents: Iterable[str], *, show_progress: bool = False) -> Vocabulary:
        """Train and return a WordPiece vocabulary from text documents."""
        basic_tokenizer = BasicTokenizer(
            do_lower_case=self.do_lower_case,
            never_split=set(SPECIAL_TOKENS),
        )
        word_counts: Counter[str] = Counter()

        with tqdm(
            documents,
            desc="Counting words",
            unit="doc",
            disable=not show_progress,
            mininterval=0.5,
        ) as document_progress:
            for document in document_progress:
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
            | set(ASCII_DIGITS)
        )
        vocabulary_tokens = [*SPECIAL_TOKENS, *initial_tokens]

        if len(vocabulary_tokens) > self.vocab_size:
            raise ValueError(
                "vocab_size is too small for special tokens "
                "and the initial character vocabulary"
            )
        
        if show_progress:
            tqdm.write(
                f"Retained words: {len(word_counts):,} "
                f"initial vocabulary: {len(vocabulary_tokens):,}"
            )
            tqdm.write("Building pair statistics and reverse indexes...")
        
        state = IncrementalWordPieceState.build(
            pieces_by_word=word_pieces,
            word_counts=word_counts,
        )

        if show_progress:
            tqdm.write(
                f"Building heap for "
                f"{len(state.global_pair_counts):,} pairs..."
            )
        
        pair_heap = PairHeap(state)
        vocabulary_set = set(vocabulary_tokens)

        with tqdm(
            total=self.vocab_size,
            initial=len(vocabulary_tokens),
            desc="Training vocabulary",
            unit="token",
            disable=not show_progress,
            mininterval=0.5,
        ) as vocabulary_progress:
            while len(vocabulary_tokens) < self.vocab_size:
                best_pair = pair_heap.pop_best()

                if best_pair is None:
                    vocabulary_progress.set_description(
                        "Finished: no pairs remain"
                    )
                    break
                
                left, right = best_pair
                new_piece = left + right.removeprefix("##")

                affected_word_count = pair_heap.merge_and_refresh(best_pair)

                if affected_word_count == 0:
                    raise RuntimeError(
                        "Heap returned a pair absent from the corpus: "
                        f"{best_pair!r}"
                    )
                
                if new_piece not in vocabulary_set:
                    vocabulary_tokens.append(new_piece)
                    vocabulary_set.add(new_piece)
                    vocabulary_progress.update(1)
        
        return Vocabulary(vocabulary_tokens)