"""Clean a raw JSONL corpus into normalized, deduplicated JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from minibert.data.corpus import clean_jsonl_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-characters", type=int, default=20)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    stats = clean_jsonl_corpus(
        input_path=args.input,
        output_path=args.output,
        min_characters=args.min_characters,
    )

    print(f"Input records: {stats.total_records}")
    print(f"Kept records: {stats.kept_records}")
    print(f"Invalid JSON records: {stats.invalid_json_records}")
    print(f"Empty records: {stats.empty_records}")
    print(f"Short records: {stats.short_records}")
    print(f"Duplicate records: {stats.duplicate_records}")

if __name__ == "__main__":
    main()