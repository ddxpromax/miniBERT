import json
from pathlib import Path

import pytest

from minibert.data.jsonl import write_text_jsonl


def test_write_text_jsonl_writes_one_text_record_per_line(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "documents.jsonl"

    stats = write_text_jsonl(
        ["First document.", "Café document."],
        output_path,
    )

    assert stats.written_records == 2
    assert [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ] == [
        {"text": "First document."},
        {"text": "Café document."},
    ]

def test_write_text_jsonl_rejects_non_string_record(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Expected str at record 2"):
        write_text_jsonl(
            ["valid", None],
            tmp_path / "documents.jsonl",
        )