"""Remove exact duplicate document IDs across corpus splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from minibert.data.deduplication import deduplicate_ordered_splits


def parse_args() -> argparse.Namespce:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    input_paths = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }
    output_paths = {
        split_name: args.output_dir / f"{split_name}.jsonl"
        for split_name in input_paths
    }

    stats = deduplicate_ordered_splits(input_paths, output_paths)

    for split_name, split_stats in stats.items():
        print(
            f"{split_name}: "
            f"input={split_stats.input_records}, "
            f"kept={split_stats.kept_records}, "
            f"removed={split_stats.removed_records}"
        )

if __name__ == "__main__":
    main()