"""Streaming readers for Parquet text corpora."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq


def iter_parquet_texts(
    input_path: Path,
    batch_size: int = 8_192,
) -> Iterator[str]:
    """Yield values from a Parquet file's required text column."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    
    parquet_file = pq.ParquetFile(input_path)

    if "text" not in parquet_file.schema_arrow.names:
        raise ValueError(f"Expected a text column in {input_path}")
    
    row_number = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["text"],
    ):
        for text in batch.column(0).to_pylist():
            row_number += 1

            if not isinstance(text, str):
                raise ValueError(
                    f"Expected a string text value in {input_path} "
                    f"at row {row_number}"
                )
            
            yield text