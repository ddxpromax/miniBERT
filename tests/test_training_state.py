from collections import Counter

from minibert.tokenizer.training_state import IncrementalWordPieceState

from minibert.tokenizer.training_state import (
    iter_pairs,
    merge_pair,
    pair_counts,
    weighted_pair_counts,
    weighted_piece_counts,
    PairIndex,
)


def test_iter_pairs_yield_adjacent_pairs() -> None:
    pieces = ("p", "##l", "##a", "##y")

    assert list(iter_pairs(pieces)) == [
        ("p", "##l"),
        ("##l", "##a"), 
        ("##a", "##y"),
    ]

def test_pair_counts_handles_repeated_pair() -> None:
    pieces = ("a", "##b", "##a", "##b")

    assert pair_counts(pieces) == Counter(
        {
            ("a", "##b"): 1,
            ("##b", "##a"): 1,
            ("##a", "##b"): 1,
        }
    )

def test_merge_pair_merges_every_non_overlapping_occurrence() -> None:
    pieces = ("a", "##b", "##a", "##b")

    assert merge_pair(pieces, ("a", "##b")) == (
        "ab",
        "##a",
        "##b",
    )

def test_weighted_piece_counts_uses_word_frequency() -> None:
    pieces_by_word = {
        "play": ("p", "##l", "##a", "##y"),
        "go": ("g", "##o"),
    }
    word_counts = Counter({"play": 3, "go": 2})

    assert weighted_piece_counts(pieces_by_word, word_counts) == Counter(
        {
            "p": 3,
            "##l": 3,
            "##a": 3,
            "##y": 3,
            "g": 2,
            "##o": 2,
        }
    )

def test_weighted_pair_counts_uses_word_frequency() -> None:
    pieces_by_word = {
        "play": ("p", "##l", "##a", "##y"),
        "go": ("g", "##o"),
    }
    word_counts = Counter({"play": 3, "go": 2})

    assert weighted_pair_counts(pieces_by_word, word_counts) == Counter(
        {
            ("p", "##l"): 3,
            ("##l", "##a"): 3,
            ("##a", "##y"): 3,
            ("g", "##o"): 2,
        }
    )

def test_pair_index_tracks_affected_words_and_pairs() -> None:
    pieces_by_word = {
        "play": ("p", "##l", "##a", "##y"),
        "played": ("p", "##l", "##a", "##y", "##e", "##d"),
        "go": ("g", "##o"),
    }

    index = PairIndex.build(pieces_by_word)

    assert index.words_for_pair(("p", "##l")) == {"play", "played"}
    assert index.words_for_pair(("g", "##o")) == {"go"}
    assert index.words_for_pair(("missing", "##pair")) == set()
    
    assert index.pairs_using_piece("##y") == {
        ("##a", "##y"),
        ("##y", "##e"),
    }
    assert index.pairs_using_piece("missing") == set()

def test_incremental_state_matches_full_recalculation_after_merge() -> None:
    pieces_by_word = {
        "play": ("p", "##l", "##a", "##y"),
        "played": ("p", "##l", "##a", "##y", "##e", "##d"),
        "go": ("g", "##o"),
    }
    word_counts = Counter({"play": 3, "played": 2, "go": 1})

    state = IncrementalWordPieceState.build(pieces_by_word, word_counts)

    assert state.merge(("p", "##l")) == 2
    assert state.pieces_by_word["play"] == ("pl", "##a", "##y")
    assert state.pieces_by_word["played"] == ("pl", "##a", "##y", "##e", "##d")
    assert state.piece_counts == weighted_piece_counts(
        state.pieces_by_word,
        word_counts,
    )
    assert state.global_pair_counts == weighted_pair_counts(
        state.pieces_by_word,
        word_counts,
    )

def test_incremental_state_updates_reverse_indexes() -> None:
    pieces_by_word = {
        "play": ("p", "##l", "##a", "##y"),
        "go": ("g", "##o"),
    }
    word_counts = Counter({"play": 1, "go": 1})

    state = IncrementalWordPieceState.build(pieces_by_word, word_counts)
    state.merge(("p", "##l"))

    assert state.index.words_for_pair(("p", "##l")) == set()
    assert state.index.words_for_pair(("pl", "##a")) == {"play"}
    assert ("p", "##l") not in state.index.pairs_using_piece("p")

def test_incremental_state_ignores_missing_pair() -> None:
    pieces_by_word = {"go": ("g", "##o")}
    word_counts = Counter({"go": 1})

    state = IncrementalWordPieceState.build(pieces_by_word, word_counts)

    assert state.merge(("missing", "##pair")) == 0
    assert state.pieces_by_word["go"] == ("g", "##o")