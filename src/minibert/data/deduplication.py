"""Exact document deduplication across ordered corpus splits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from minibert.data.corpus import iter_jsonl_records


@dataclass
class SplitDeduplicationStats:
    """Counts for one split after exact cross-split deduplication."""

    input_records: int = 0
    kept_records: int = 0
    removed_records: int = 0

def deduplicate_ordered_splits(
    input_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
) -> dict[str, SplitDeduplicationStats]:
    """Write disjoint split files, preserving the order of input_paths.
    
    An ID first seen in an earlier split is retained. Any later occurrence,
    including an in-split duplicate, is omitted.
    """
    if not input_paths:
        raise ValueError("input_paths must not be empty")

    if set(input_paths) != set(output_paths):
        raise ValueError("input_paths and output_paths must have identical keys")
    
    for split_name, input_path in input_paths.items():
        if input_path.resolve() == output_paths[split_name].resolve():
            raise ValueError(
                f"Input and output paths must differ for split {split_name!r}"
            )
    
    seen_ids: set[str] = set()
    all_stats: dict[str, SplitDeduplicationStats] = {}

    for split_name, input_path in input_paths.items():
        output_path = output_paths[split_name]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = SplitDeduplicationStats()

        with output_path.open("w", encoding="utf-8") as output_file:
            for line_number, record in iter_jsonl_records(input_path):
                stats.input_records += 1

                identifier = record.get("id")
                if not isinstance(identifier, str):
                    raise ValueError(
                        f"Expected a string id in {input_path }"
                        f"at line {line_number}"
                    )
                
                if identifier in seen_ids:
                    stats.removed_records += 1
                    continue
                
                seen_ids.add(identifier)
                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                stats.kept_records += 1

        all_stats[split_name] = stats
    
    return all_stats