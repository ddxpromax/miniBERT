"""Validation helpers for pretraining corpus splites."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from minibert.data.corpus import iter_jsonl_records


def validate_disjoint_splits(split_paths: Mapping[str, Path]) -> dict[str, int]:
    """Ensure every JSONL record ID occurs in exactly one named split."""
    if not split_paths:
        raise ValueError("split_paths must not be empty")
    
    seen_ids: dict[str, str] = {}
    record_counts: dict[str, int] = {}

    for split_name, input_path in split_paths.items():
        count = 0

        for line_number, record in iter_jsonl_records(input_path):
            identifier = record.get("id")

            if not isinstance(identifier, str):
                raise ValueError(
                    f"Expected a string id in {input_path} "
                    f"at line {line_number}"
                )
            
            previous_split = seen_ids.get(identifier)
            if previous_split is not None:
                raise ValueError(
                    f"Duplicate document ID found in both "
                    f"{previous_split!r} and {split_name!r}"
                )
            
            seen_ids[identifier] = split_name
            count += 1
        
        record_counts[split_name] = count
    
    return record_counts