"""Pure data operations used by the incremental WordPiece trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from collections.abc import Iterable

Pair = tuple[str, str]
PieceSequence = tuple[str, ...]


def iter_pairs(pieces: PieceSequence) -> Iterable[Pair]:
    """Yield every adjacent WordPiece pair in one piece sequence."""
    for index in range(len(pieces) - 1):
        yield pieces[index], pieces[index + 1]

def pair_counts(pieces: PieceSequence) -> Counter[Pair]:
    """Count adjacent pairs within one piece sequence."""
    return Counter(iter_pairs(pieces))

def merge_pair(pieces: PieceSequence, pair: Pair) -> PieceSequence:
    """Merge non-overlapping occurrences of one pair in a sequence."""
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

def weighted_piece_counts(
    pieces_by_word: dict[str, PieceSequence],
    word_counts: Counter[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()

    for word, pieces in pieces_by_word.items():
        frequency = word_counts[word]

        for piece in pieces:
            counts[piece] += frequency
    
    return counts

def weighted_pair_counts(
    pieces_by_word: dict[str, PieceSequence],
    word_counts: Counter[str],
) -> Counter[Pair]:
    """Count adjacent pairs, weighting each word by corpus frequency."""
    counts: Counter[Pair] = Counter()

    for word, pieces in pieces_by_word.items():
        frequency = word_counts[word]

        for pair, occurrences in pair_counts(pieces).items():
            counts[pair] += frequency * occurrences
        
    return counts

@dataclass
class PairIndex:
    """Reverse indexes required for incremental WordPiece training."""

    pair_to_words: dict[Pair, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    piece_to_pairs: dict[str, set[Pair]] = field(
        default_factory=lambda: defaultdict(set)
    )

    @classmethod
    def build(
        cls,
        pieces_by_word: dict[str, PieceSequence],
    ) -> "PairIndex":
        """Build reverse indexes from current word-piece segmentations."""
        index = cls()

        for word, pieces in pieces_by_word.items():
            for pair in set(iter_pairs(pieces)):
                left, right = pair
                index.pair_to_words[pair].add(word)
                index.piece_to_pairs[left].add(pair)
                index.piece_to_pairs[right].add(pair)

        return index
    
    def words_for_pair(self, pair: Pair) -> set[str]:
        """Return words currently containing a pair."""
        return self.pair_to_words.get(pair, set())
    
    def pairs_using_piece(self, piece: str) -> set[Pair]:
        """Return pairs whose score depends on one piece's frequency."""
        return self.piece_to_pairs.get(piece, set())

@dataclass
class IncrementalWordPieceState:
    """Mutable WordPiece state with incrementally maintained statistics."""

    pieces_by_word: dict[str, PieceSequence]
    word_counts: Counter[str]
    piece_counts: Counter[str]
    global_pair_counts: Counter[Pair]
    index: PairIndex

    @classmethod
    def build(
        cls,
        pieces_by_word: dict[str, PieceSequence],
        word_counts: Counter[str],
    ) -> "IncrementalWordPieceState":
        """Create state and calculate its initial global statistics."""
        return cls(
            pieces_by_word=pieces_by_word,
            word_counts=word_counts,
            piece_counts=weighted_piece_counts(pieces_by_word, word_counts),
            global_pair_counts=weighted_pair_counts(
                pieces_by_word,
                word_counts,
            ),
            index=PairIndex.build(pieces_by_word),
        )
    
    @staticmethod
    def _adjust_count(
        counts: Counter[str] | Counter[Pair],
        key: str | Pair,
        amount: int,
    ) -> None:
        """Add an amount and remove the key when its count reaches zero."""
        counts[key] += amount

        if counts[key] == 0:
            del counts[key]
    
    def merge(self, pair: Pair) -> int:
        """Merge one pair in all affected words and return affected word count."""
        affected_words = list(self.index.words_for_pair(pair))

        for word in affected_words:
            old_pieces = self.pieces_by_word[word]
            new_pieces = merge_pair(old_pieces, pair)
            self._replace_word_pieces(word, old_pieces, new_pieces)

        return len(affected_words)

    def _replace_word_pieces(
        self, 
        word: str,
        old_pieces: PieceSequence,
        new_pieces: PieceSequence,
    ) -> None:
        """Replace one word segmentation and incrementally update all counts."""
        frequency = self.word_counts[word]

        old_piece_counts = Counter(old_pieces)
        new_piece_counts = Counter(new_pieces)

        for piece in set(old_piece_counts) | set(new_piece_counts):
            difference = new_piece_counts[piece] - old_piece_counts[piece]

            if difference:
                self._adjust_count(
                    self.piece_counts,
                    piece,
                    difference * frequency,
                )
        
        old_pair_counts = pair_counts(old_pieces)
        new_pair_counts = pair_counts(new_pieces)

        for pair in set(old_pair_counts) | set(new_pair_counts):
            difference = new_pair_counts[pair] - old_pair_counts[pair]

            if not difference:
                continue
            
            previous_global_count = self.global_pair_counts.get(pair, 0)

            self._adjust_count(
                self.global_pair_counts,
                pair,
                difference * frequency,
            )

            current_global_count = self.global_pair_counts.get(pair, 0)
            left, right = pair

            if previous_global_count == 0 and current_global_count > 0:
                self.index.piece_to_pairs[left].add(pair)
                self.index.piece_to_pairs[right].add(pair)

            if previous_global_count > 0 and current_global_count == 0:
                self.index.piece_to_pairs[left].discard(pair)
                self.index.piece_to_pairs[right].discard(pair)
            
        for pair in set(old_pair_counts) - set(new_pair_counts):
            self.index.pair_to_words[pair].discard(word)
        
        for pair in set(new_pair_counts) - set(old_pair_counts):
            self.index.pair_to_words[pair].add(word)

        self.pieces_by_word[word] = new_pieces