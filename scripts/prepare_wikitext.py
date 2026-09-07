"""Convert raw WikiText Parquet shards into article-level JSONL."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from minibert.data.jsonl import write_text_jsonl
from minibert.data.parquet import iter_parquet_texts
from minibert.data.wikitext import iter_wikitext_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8_192)
    return parser.parse_args()

def iter_shard_lines(
    input_paths: list[Path],
    batch_size: int,
) -> Iterator[str]:
    """Yield text rows from shards in the order supplied."""
    for input_path in input_paths:
        yield from iter_parquet_texts(input_path, batch_size=batch_size)

def main() -> None:
    args = parse_args()

    lines = iter_shard_lines(args.inputs, args.batch_size)
    documents = iter_wikitext_documents(lines)
    stats = write_text_jsonl(documents, args.output)

    print(f"Output: {args.output}")
    print(f"Written documents: {stats.written_records}")

if __name__ == "__main__":
    main()