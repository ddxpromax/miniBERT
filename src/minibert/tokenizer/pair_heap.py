"""Heap construction for WordPiece pair selection."""

from __future__ import annotations

from collections.abc import Iterable

import heapq

from minibert.tokenizer.training_state import (
    IncrementalWordPieceState,
    Pair,
)

PairHeapEntry = tuple[float, int, Pair]
VersionedPairHeapEntry = tuple[float, int, Pair, int]


def build_pair_heap(
    state: IncrementalWordPieceState,
) -> list[PairHeapEntry]:
    """Build a heap from the state's current pair scores."""
    entries: list[PairHeapEntry] = []

    for pair, pair_frequency in state.global_pair_counts.items():
        if pair_frequency <= 0:
            continue
        
        left, right = pair
        left_frequency = state.piece_counts[left]
        right_frequency = state.piece_counts[right]

        if left_frequency <= 0 or right_frequency <= 0:
            raise ValueError(
                f"Positive pair count requires positive piece counts: {pair!r}"
            )
        
        score = pair_frequency / (left_frequency * right_frequency)
        entries.append((-score, -pair_frequency, pair))

    heapq.heapify(entries)
    return entries

class PairHeap:
    """Pair priority queue with explicit refresh and lazy invalidtion."""

    def __init__(self, state: IncrementalWordPieceState) -> None:
        self._state = state

        initial_entries = build_pair_heap(state)

        self._versions: dict[Pair, int] = {pair: 0 for _, _, pair in initial_entries}
        self._heap: list[VersionedPairHeapEntry] = [
            (negative_score, negative_frequency, pair, 0)
            for negative_score, negative_frequency, pair in initial_entries
        ]
    
    def refresh_pairs(self, pairs: Iterable[Pair]) -> None:
        """Refresh supplied pairs after the state's statistics change."""
        for pair in set(pairs):
            self._refresh_pair(pair)
        
        self._maybe_compact()
    
    def _refresh_pair(self, pair: Pair) -> None:
        frequency = self._state.global_pair_counts.get(pair, 0)
        version = self._versions.get(pair, 0) + 1

        if frequency <= 0:
            self._versions[pair] = version
            return
        
        left, right = pair
        left_frequency = self._state.piece_counts[left]
        right_frequency = self._state.piece_counts[right]

        if left_frequency <= 0 or right_frequency <= 0:
            raise ValueError(
                f"Positive pair count requires positive piece counts: {pair!r}"
            )

        score = frequency / (left_frequency * right_frequency)
        
        self._versions[pair] = version
        heapq.heappush(
            self._heap,
            (-score, -frequency, pair, version),
        )
    
    def pop_best(self) -> Pair | None:
        """Remove and return the best current candidate, or None."""
        while self._heap:
            _, _, pair, version = heapq.heappop(self._heap)

            if version != self._versions.get(pair):
                continue
        
            return pair
        
        return None
    
    def merge_and_refresh(self, pair: Pair) -> int:
        """Merge a pair, refresh affected candidates, 
        and return word count.
        """
        left, right = pair
        merged_piece = left + right.removeprefix("##")

        pairs_to_refresh = set(
            self._state.index.pairs_using_piece(left)
        )
        pairs_to_refresh.update(
            self._state.index.pairs_using_piece(right)
        )

        affected_word_count = self._state.merge(pair)

        if affected_word_count == 0:
            return 0
        
        pairs_to_refresh.update(
            self._state.index.pairs_using_piece(merged_piece)
        )

        self.refresh_pairs(pairs_to_refresh)

        return affected_word_count

    def compact(self) -> None:
        """Remove stale entries and unused version records."""
        self._heap = [
            entry
            for entry in self._heap
            if entry[3] == self._versions.get(entry[2])
        ]

        heapq.heapify(self._heap)

        self._versions = {
            pair: version
            for _, _, pair, version in 
            self._heap
        }

    def _maybe_compact(self) -> None:
        """Compact when stored entries greatly exceed current pair count."""
        current_pair_count = len(self._state.global_pair_counts)
        threshold = max(1_024, 4 * current_pair_count)

        if (
            len(self._heap) > threshold
            or len(self._versions) > threshold
        ):
            self.compact()