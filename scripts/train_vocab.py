"""Train a WordPiece vocabulary from a cleaned JSONL corpus."""

from __future__ import annotations

from time import perf_counter

import argparse
from pathlib import Path

from minibert.data.corpus import iter_corpus_texts
from minibert.tokenizer.trainer import WordPieceTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--no-progress", action="store_true", help="Disable training progress output.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        do_lower_case=True,
        min_frequency=args.min_frequency,
    )

    started_at = perf_counter()

    vocabulary = trainer.train(
        iter_corpus_texts(args.input),
        show_progress=not args.no_progress,
    )

    training_seconds = perf_counter() - started_at
    vocabulary.save(args.output)

    print(f"Training time: {training_seconds:.1f} seconds")

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print("First 20 tokens:")
    for identifier, token in enumerate(vocabulary.tokens[:20]):
        print(f"{identifier:>5}: {token}")

if __name__ == "__main__":
    main()