import heapq
from collections import Counter

import pytest

from minibert.tokenizer.pair_heap import build_pair_heap, PairHeap
from minibert.tokenizer.training_state import IncrementalWordPieceState


def test_pair_heap_orders_by_score_then_lexicographic_order() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={
            "ab": ("a", "##b"),
            "db": ("d", "##b"),
            "ac": ("a", "##c"),
        },
        word_counts=Counter({"ab": 1, "db": 1, "ac": 1}),
    )

    pair_heap = build_pair_heap(state)

    assert len(pair_heap) == 3

    entries = [heapq.heappop(pair_heap) for _ in range(3)]
    assert entries == [
        (-0.5, -1, ("a", "##c")),
        (-0.5, -1, ("d", "##b")),
        (-0.25, -1, ("a", "##b")),
    ]
    assert pair_heap == []

def test_pair_heap_prefers_higher_frequency_when_scores_are_equal() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={
            "ab": ("a", "##b"),
            "a": ("a",),
            "cd": ("c", "##d"),
        },
        word_counts=Counter({"ab": 1, "a": 1, "cd": 2}),
    )

    pair_heap = build_pair_heap(state)

    assert heapq.heappop(pair_heap) == (-0.5, -2, ("c", "##d"))
    assert heapq.heappop(pair_heap) == (-0.5, -1, ("a", "##b"))

def test_pair_heap_returns_empty_list_when_no_pair_exist() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={"a": ("a",)},
        word_counts=Counter({"a": 3}),
    )

    assert build_pair_heap(state) == []

@pytest.mark.parametrize("piece", ["a", "##b"])
def test_pair_heap_rejects_nonpositive_piece_frequency(piece: str) -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={"ab": ("a", "##b")},
        word_counts=Counter({"ab": 1}),
    )

    state.piece_counts[piece] = 0

    with pytest.raises(
        ValueError,
        match="Positive pair count requires positive piece counts",
    ):
        build_pair_heap(state)

def test_versioned_heap_skips_old_entries_after_repeated_refresh() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={"ab": ("a", "##b")},
        word_counts=Counter({"ab": 1}),
    )
    pair_heap = PairHeap(state)
    pair = ("a", "##b")

    pair_heap.refresh_pairs([pair])
    pair_heap.refresh_pairs([pair])

    assert pair_heap.pop_best() == pair
    assert pair_heap.pop_best() is None

def test_versioned_heap_invalidates_removed_pair() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={"ab": ("a", "##b")},
        word_counts=Counter({"ab": 1}),
    )
    pair_heap = PairHeap(state)
    pair = ("a", "##b")

    state.merge(pair)
    pair_heap.refresh_pairs([pair])

    assert pair_heap.pop_best() is None

def test_versioned_heap_add_new_pair_after_merge() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={"abc": ("a", "##b", "##c")},
        word_counts=Counter({"abc": 1}),
    )
    pair_heap = PairHeap(state)

    state.merge(("a", "##b"))
    pair_heap.refresh_pairs(
        [
            ("a", "##b"),
            ("##b", "##c"),
            ("ab", "##c"),
        ]
    )

    assert pair_heap.pop_best() == ("ab", "##c")
    assert pair_heap.pop_best() is None

def test_versioned_heap_refreshes_score_when_piece_frequency_changes() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={
            "ab": ("a", "##b"),
            "ac": ("a", "##c"),
            "db": ("d", "##b"),
        },
        word_counts=Counter({"ab": 1, "ac": 1, "db": 1}),
    )
    pair_heap = PairHeap(state)

    assert pair_heap.pop_best() == ("a", "##c")

    state.merge(("a", "##c"))

    assert state.global_pair_counts[("a", "##b")] == 1
    assert state.piece_counts["a"] == 1

    pair_heap.refresh_pairs(
        [
            ("a", "##c"),
            ("a", "##b"),
        ]
    )

    assert pair_heap.pop_best() == ("a", "##b")
    assert pair_heap.pop_best() == ("d", "##b")
    assert pair_heap.pop_best() is None

def test_merge_and_refresh_replaces_old_pairs_with_new_pair() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={"abc": ("a", "##b", "##c")},
        word_counts=Counter({"abc": 3}),
    )
    pair_heap = PairHeap(state)

    affected_word_count = pair_heap.merge_and_refresh(("a", "##b"))
    
    assert affected_word_count == 1
    assert state.pieces_by_word["abc"] == ("ab", "##c")
    assert pair_heap.pop_best() == ("ab", "##c")
    assert pair_heap.pop_best() is None

def test_merge_and_refresh_updates_pairs_using_old_piece() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={
            "ab": ("a", "##b"),
            "ac": ("a", "##c"),
            "db": ("d", "##b"),
        },
        word_counts=Counter({"ab": 1, "ac": 1, "db": 1}),
    )
    pair_heap = PairHeap(state)

    assert pair_heap.pop_best() == ("a", "##c")
    assert pair_heap.merge_and_refresh(("a", "##c")) == 1
    
    assert state.global_pair_counts[("a", "##b")] == 1
    assert pair_heap.pop_best() == ("a", "##b")
    assert pair_heap.pop_best() == ("d", "##b")
    assert pair_heap.pop_best() is None

def test_merge_and_refresh_updates_pairs_using_existing_merged_piece() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={
            "abc": ("a", "##b", "##c"),
            "abd": ("ab", "##d"),
            "xy": ("x", "##y"),
        },
        word_counts=Counter({"abc": 1, "abd": 1, "xy": 2}),
    )
    pair_heap = PairHeap(state)

    pair_heap.merge_and_refresh(("a", "##b"))

    assert state.piece_counts["ab"] == 2

    assert pair_heap.pop_best() == ("x", "##y")
    assert pair_heap.pop_best() == ("ab", "##c")
    assert pair_heap.pop_best() == ("ab", "##d")
    assert pair_heap.pop_best() is None

def test_merge_and_refresh_matched_rebuilt_heap_across_merges() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={
            "play": ("p", "##l", "##a", "##y"),
            "played": ("p", "##l", "##a", "##y", "##e", "##d"),
            "aaaa": ("a", "##a", "##a", "##a"),
            "go": ("g", "##o"),
        },
        word_counts=Counter({
            "play": 3,
            "played": 2,
            "aaaa": 4,
            "go": 1,
        }),
    )
    pair_heap = PairHeap(state)
    
    while state.global_pair_counts:
        rebuilt_heap = build_pair_heap(state)
        _, _, expected_pair = heapq.heappop(rebuilt_heap)

        best_pair = pair_heap.pop_best()

        assert best_pair == expected_pair
        assert pair_heap.merge_and_refresh(best_pair) > 0
    
    assert pair_heap.pop_best() is None

def test_compact_preserves_candidate_order_and_removes_stale_entries() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={
            "ab": ("a", "##b"),
            "ac": ("a", "##c"),
            "db": ("d", "##b"),
        },
        word_counts=Counter({"ab": 1, "ac": 1, "db": 1}),
    )
    pair_heap = PairHeap(state)

    for _ in range(5):
        pair_heap.refresh_pairs(state.global_pair_counts)

    pair_heap.compact()

    assert len(pair_heap._heap) == 3
    assert len(pair_heap._versions) == 3
    
    assert pair_heap.pop_best() == ("a", "##c")
    assert pair_heap.pop_best() == ("d", "##b")
    assert pair_heap.pop_best() == ("a", "##b")
    assert pair_heap.pop_best() is None

def test_compact_does_not_restore_popped_candidate() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={"ab": ("a", "##b")},
        word_counts=Counter({"ab": 1}),
    )
    pair_heap = PairHeap(state)

    pair_heap.refresh_pairs([("a", "##b")])
    assert pair_heap.pop_best() == ("a", "##b")

    pair_heap.compact()
    assert pair_heap._heap == []
    assert pair_heap._versions == {}
    assert pair_heap.pop_best() is None

    pair_heap.refresh_pairs([("a", "##b")])
    assert pair_heap.pop_best() == ("a", "##b")
    assert pair_heap.pop_best() is None

def test_refresh_automatically_limits_stale_entry_storage() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={"ab": ("a", "##b")},
        word_counts=Counter({"ab": 1}),
    )
    pair_heap = PairHeap(state)

    for _ in range(1_100):
        pair_heap.refresh_pairs([("a", "##b")])
    
    assert len(pair_heap._heap) <= 1_024
    assert pair_heap.pop_best() == ("a", "##b")
    assert pair_heap.pop_best() is None

def test_refresh_automatically_cleans_unused_version_records() -> None:
    state = IncrementalWordPieceState.build(
        pieces_by_word={"ab": ("a", "##b")},
        word_counts=Counter({"ab": 1}),
    )
    pair_heap = PairHeap(state)

    pair_heap.refresh_pairs(
        ("missing", f"##{index}")
        for index in range(1_100)
    )
    
    assert len(pair_heap._versions) == 1
    assert pair_heap.pop_best() == ("a", "##b")
    assert pair_heap.pop_best() is None