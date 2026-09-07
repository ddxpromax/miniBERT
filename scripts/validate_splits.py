"""Verify that processed corpus splits are pairwise disjoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from minibert.data.splits import validate_disjoint_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    counts = validate_disjoint_splits(
        {
            "train": args.train,
            "validation": args.validation,
            "test": args.test,
        }
    )

    for split_name, count in counts.items():
        print(f"{split_name}: {count} documents")

    print("Split validation passed: no duplicate document IDs found.")

if __name__ == "__main__":
    main()