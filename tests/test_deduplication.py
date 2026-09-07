import json
from pathlib import Path

import pytest

from minibert.data.deduplication import deduplicate_ordered_splits


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

def read_jsonl(path: Path) -> list[dict[str, str]]: 
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

def test_deduplication_keeps_earlier_split_records(tmp_path: Path) -> None:
    input_paths = {
        "train": tmp_path / "train_input.jsonl",
        "validation": tmp_path / "validation_input.jsonl",
        "test": tmp_path / "test_input.jsonl",
    }
    output_paths = {
        "train": tmp_path / "output" / "train.jsonl",
        "validation": tmp_path / "output" / "validation.jsonl",
        "test": tmp_path / "output" / "test.jsonl",
    }

    write_jsonl(
        input_paths["train"],
        [
            {"id": "a", "text": "Train A."},
            {"id": "b", "text": "Train B."},
        ],
    )
    write_jsonl(
        input_paths["validation"],
        [
            {"id": "b", "text": "Duplicate of train B."},
            {"id": "c", "text": "Validation C."},
        ],
    )
    write_jsonl(
        input_paths["test"],
        [
            {"id": "a", "text": "Duplicate of train A."},
            {"id": "c", "text": "Duplicate of validation C."},
            {"id": "d", "text": "Test D."},
        ],
    )

    stats = deduplicate_ordered_splits(input_paths, output_paths)

    assert stats["train"].input_records == 2
    assert stats["train"].kept_records == 2
    assert stats["train"].removed_records == 0

    assert stats["validation"].kept_records == 1
    assert stats["validation"].removed_records == 1

    assert stats["test"].kept_records == 1
    assert stats["test"].removed_records == 2

    assert read_jsonl(output_paths["train"]) == [
        {"id": "a", "text": "Train A."},
        {"id": "b", "text": "Train B."},
    ]
    assert read_jsonl(output_paths["validation"]) == [
        {"id": "c", "text": "Validation C."},
    ]
    assert read_jsonl(output_paths["test"]) == [
        {"id": "d", "text": "Test D."},
    ]

def test_deduplication_rejects_mismatched_split_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identical keys"):
        deduplicate_ordered_splits(
            {"train": tmp_path / "train.jsonl"},
            {"validation": tmp_path / "validation.jsonl"},
        )

def test_deduplication_rejects_same_input_and_output_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.jsonl"

    with pytest.raises(ValueError, match="Input and output paths must differ"):
        deduplicate_ordered_splits(
            {"train": path},
            {"train": path},
        )