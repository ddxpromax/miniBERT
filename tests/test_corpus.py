import json
from pathlib import Path

import pytest

from minibert.data.corpus import clean_jsonl_corpus, document_id, iter_jsonl_records


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

def read_jsonl(path: Path) -> list[dict[str, str]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

def test_document_id_is_deterministic() -> None:
    assert document_id("hello") == document_id("hello")
    assert document_id("hello") != document_id("Hello")
    assert len(document_id("hello")) == 64

def test_clean_jsonl_corpus_normalizes_filters_and_deduplicates(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "clean.jsonl"

    write_text(
        input_path, 
        "\n".join(
            [
                '{"text": "  A sufficiently long\\tfirst document.  "}',
                '{"text": "A sufficiently long first document."}',
                '{"text": "too short"}',
                '{"text": "  \\t\\n  "}',
                '{"url": "https://example.com"}',
            ]
        )
        + "\n",
    )

    stats = clean_jsonl_corpus(input_path, output_path, min_characters=20)

    assert stats.total_records == 5
    assert stats.kept_records == 1
    assert stats.duplicate_records == 1
    assert stats.short_records == 1
    assert stats.empty_records == 1
    assert stats.invalid_schema_records == 1

    records = read_jsonl(output_path)
    assert records == [
        {
            "id": document_id("A sufficiently long first document."),
            "text": "A sufficiently long first document.",
        }
    ]

def test_clean_jsonl_corpus_rejects_invalid_json(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "clean.jsonl"
    write_text(input_path, '{"text": "valid"}\nnot json\n')

    with pytest.raises(ValueError, match="Invalid JSON.*line 2"):
        clean_jsonl_corpus(input_path, output_path)
    
def test_clean_jsonl_corpus_rejects_invalid_minimum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_characters must be at least 1"):
        clean_jsonl_corpus(
            tmp_path / "input.jsonl",
            tmp_path / "output.jsonl",
            min_characters=0,
        )

def test_iter_jsonl_records_rejects_non_object_json(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    write_text(input_path, '["not", "an object"]\n')

    with pytest.raises(ValueError, match="Expected a JSON object.*line 1"):
        list(iter_jsonl_records(input_path))