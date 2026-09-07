from pathlib import Path

import pytest

from minibert.data.splits import validate_disjoint_splits


def write_jsonl(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

def test_validate_disjoint_splits_return_document_counts(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"

    write_jsonl(
        train_path,
        [
            '{"id": "train-one", "text": "First train document."}',
            '{"id": "train-two", "text": "Second train document."}',
        ],
    )
    write_jsonl(
        validation_path, 
        [
            '{"id": "validation-one", "text": "Validation document."}',
        ]
    )

    assert validate_disjoint_splits(
        {"train": train_path, "validation": validation_path}
    ) == {"train": 2, "validation": 1}

def test_validate_disjoint_splits_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"

    write_jsonl(train_path, ['{"id": "same", "text": "Train."}'])
    write_jsonl(validation_path, ['{"id": "same", "text": "Validation."}'])

    with pytest.raises(ValueError, match="both 'train' and 'validation'"):
        validate_disjoint_splits(
            {"train": train_path, "validation": validation_path}
        )

def test_validate_disjoint_splits_rejects_missing_id(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    write_jsonl(train_path, ['{"text": "No ID"}'])

    with pytest.raises(ValueError, match="Expected a string id.*line 1"):
        validate_disjoint_splits({"train": train_path})

def test_validate_disjoint_splits_rejects_empty_mapping() -> None:
    with pytest.raises(ValueError, match="split_paths must not be empty"):
        validate_disjoint_splits({})