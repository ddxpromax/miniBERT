from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from minibert.data.parquet import iter_parquet_texts


def write_parquet(path: Path, columns: dict[str, list[object]]) -> None:
    table = pa.Table.from_pydict(columns)
    pq.write_table(table, path)

def test_iter_parquet_texts_reads_in_batches(tmp_path: Path) -> None:
    input_path = tmp_path / "documents.parquet"
    write_parquet(
        input_path,
        {"text": ["First.", "Second.", "Third."]},
    )

    assert list(iter_parquet_texts(input_path, batch_size=2)) == [
        "First.",
        "Second.",
        "Third.",
    ]

def test_iter_parquet_texts_rejects_missing_text_column(tmp_path: Path) -> None:
    input_path = tmp_path / "documents.parquet"
    write_parquet(input_path, {"content": ["First."]})

    with pytest.raises(ValueError, match="Expected a text column"):
        list(iter_parquet_texts(input_path))

def test_iter_parquet_texts_rejects_non_string_value(tmp_path: Path) -> None:
    input_path = tmp_path / "documents.parquet"
    write_parquet(input_path, {"text": [123]})

    with pytest.raises(ValueError, match="Expected a string text value.*row 1"):
        list(iter_parquet_texts(input_path))

def test_iter_parquet_texts_rejects_invalid_batch_size(tmp_path: Path) -> None:
    input_path = tmp_path / "documents.parquet"
    write_parquet(input_path, {"text": ["First."]})

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        list(iter_parquet_texts(input_path, batch_size=0))