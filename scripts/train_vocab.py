"""Train a WordPiece vocabulary from a cleaned JSONL corpus."""

from __future__ import annotations

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
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        do_lower_case=True,
        min_frequency=args.min_frequency,
    )
    vocabulary = trainer.train(iter_corpus_texts(args.input))
    vocabulary.save(args.output)

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print("First 20 tokens:")
    for identifier, token in enumerate(vocabulary.tokens[:20]):
        print(f"{identifier:>5}: {token}")

if __name__ == "__main__":
    main()