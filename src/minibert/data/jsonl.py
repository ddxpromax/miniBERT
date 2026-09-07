"""Writers for JSONL text corpora."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class JsonlWriteStats:
    """Statistics produced while writing text records."""

    written_records: int = 0

def write_text_jsonl(
    texts: Iterable[str],
    output_path: Path,
) -> JsonlWriteStats:
    """Write text strings as one JSON object per line."""
    stats = JsonlWriteStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record_number, text in enumerate(texts, start=1):
            if not isinstance(text, str):
                raise TypeError(
                    f"Expected str at record {record_number}, "
                    f"got {type(text).__name__}"
                )
            
            output_file.write(
                json.dumps({"text": text}, ensure_ascii=False) + "\n"
            )
            stats.written_records += 1
    
    return stats