"""Streaming utilities for cleaning JSONL text corpora."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

from minibert.data.normalization import normalize_document


@dataclass
class CleaningStats:
    """Counters collected while cleaning one corpus file."""

    total_records: int = 0
    kept_records: int = 0
    invalid_json_records: int = 0
    invalid_schema_records: int = 0
    empty_records: int = 0
    short_records: int = 0
    duplicate_records: int = 0

def document_id(text: str) -> str:
    """Return the derterministic SHA-256 identifier for normalized text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def iter_jsonl_records(input_path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield non-empty JSON-object records with their one-based line numbers."""
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {input_path} at line {line_number}"
                ) from error
            
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {input_path} at line {line_number}"
                )
            
            yield line_number, record

def clean_jsonl_corpus(
    input_path: Path,
    output_path: Path, 
    min_characters: int = 20,
) -> CleaningStats:
    """Normalize, filter, and exactly deduplicate a JSON text corpus.

    Each input record must have a string field named ``text``. Output records
    contain only a derterministic ``id`` and normalized ``text`` field.
    """
    if min_characters < 1:
        raise ValueError("min_characters must be at least 1")
    
    stats = CleaningStats()
    seen_ids: set[str] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for _, record in iter_jsonl_records(input_path):
            stats.total_records += 1

            raw_text =record.get("text")
            if not isinstance(raw_text, str):
                stats.invalid_schema_records += 1
                continue
            
            text = normalize_document(raw_text)
            if text is None:
                stats.empty_records += 1
                continue
            
            if len(text) < min_characters:
                stats.short_records += 1
                continue
            
            identifier = document_id(text)
            if identifier in seen_ids:
                stats.duplicate_records += 1
                continue
            
            seen_ids.add(identifier)

            output_record = {"id": identifier, "text": text}
            output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            stats.kept_records += 1
    
    return stats