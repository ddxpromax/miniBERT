"""Compact storage for tokenized documents."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from minibert.tokenizer.vocabulary import Vocabulary

TOKEN_DTYPE = np.dtype("<u2")
OFFSET_DTYPE = np.dtype("<u8")


def write_tokenized_corpus(
    documents: Iterable[list[int]],
    output_dir: Path,
    vocabulary: Vocabulary,
) -> None:
    """Write document IDs and boundaries into a new directory.
    
    Empty documents are skipped. Existing directories are never 
    overwritten. Metadata is written last to mark completion.
    """
    if len(vocabulary) > 65536:
        raise ValueError("uint16 storage supports at most 65536 tokens")
    
    output_dir.mkdir(parents=True, exist_ok=False)

    offsets = [0]

    with (output_dir / "tokens.bin").open("wb") as token_file:
        for token_ids in documents:
            if not token_ids:
                continue

            if any(
                not isinstance(identifier, int)
                or isinstance(identifier, bool)
                or not 0 <= identifier < len(vocabulary)
                for identifier in token_ids
            ):
                raise ValueError(
                    "token IDs must be integers within vocabulary range"
                )
            
            array = np.asarray(token_ids, dtype=TOKEN_DTYPE)
            array.tofile(token_file)

            offsets.append(offsets[-1] + len(token_ids))
    
    np.save(
        output_dir / "offsets.npy",
        np.asarray(offsets, dtype=OFFSET_DTYPE),
        allow_pickle=False,
    )

    vocabulary.save(output_dir / "vocab.txt")

    metadata = {
        "format_version": 1,
        "token_dtype": TOKEN_DTYPE.str,
        "offset_dtype": OFFSET_DTYPE.str,
        "vocab_size": len(vocabulary),
        "num_documents": len(offsets) - 1,
        "num_tokens": offsets[-1],
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

class TokenizedCorpus:
    """Read tokenized documents using a memory-mapped token file."""
    
    def __init__(self, directory: Path) -> None:
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )

        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")

        if (
            metadata.get("format_version") != 1
            or metadata.get("token_dtype") != TOKEN_DTYPE.str
            or metadata.get("offset_dtype") != OFFSET_DTYPE.str
        ):
            raise ValueError("unsupported corpus format")
        
        for name in ("num_documents", "num_tokens", "vocab_size"):
            value = metadata.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid metadata field: {name}")
            
        self.num_documents = metadata["num_documents"]
        self.num_tokens = metadata["num_tokens"]

        self.vocabulary = Vocabulary.from_file(directory / "vocab.txt")

        if (
            len(self.vocabulary) != metadata["vocab_size"]
            or len(self.vocabulary) > 65536
        ):
            raise ValueError("vocabulary size does not match storage")
        
        self._offsets = np.load(
            directory / "offsets.npy",
            allow_pickle=False,
        )

        if (
            self._offsets.dtype != OFFSET_DTYPE
            or self._offsets.shape != (self.num_documents + 1,)
        ):
            raise ValueError("invalid offsets array")
        
        if (
            self._offsets[0] != 0
            or self._offsets[-1] != self.num_tokens
            or np.any(self._offsets[1:] <= self._offsets[:-1])
        ):
            raise ValueError("invalid document boundaries")
        
        token_path = directory / "tokens.bin"
        expected_bytes = self.num_tokens * TOKEN_DTYPE.itemsize

        if token_path.stat().st_size != expected_bytes:
            raise ValueError("token file size does not match metadata")
        
        if self.num_tokens == 0:
            self._tokens = np.empty(0, dtype=TOKEN_DTYPE)
        else:
            self._tokens = np.memmap(
                token_path,
                dtype=TOKEN_DTYPE,
                mode="r",
                shape=(self.num_tokens,),
            )
        
    def __len__(self) -> int:
        return self.num_documents
    
    def __getitem__(self, index: int) -> np.ndarray:
        if type(index) is not int:
            raise TypeError("document index must be an integer")
        
        if index < 0 or index >= self.num_documents:
            raise IndexError("document index out of range")
        
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])

        token_ids = self._tokens[start:end].astype(np.int64, copy=True)

        if np.any(token_ids >= len(self.vocabulary)):
            raise ValueError("stored token ID is outside vocabulary range")
        
        return token_ids