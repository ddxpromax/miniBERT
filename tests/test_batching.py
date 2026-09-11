import numpy as np
import pytest

from minibert.data.batching import PaddedBatch, iter_padded_batches


def test_batch_pads_to_longest_sequence() -> None:
    sequences = [
        np.asarray([2, 5, 3], dtype=np.int64),
        np.asarray([2, 6, 7, 8, 3], dtype=np.int64),
    ]

    batches = list(iter_padded_batches(sequences, batch_size=2))

    assert len(batches) == 1

    batch = batches[0]

    assert isinstance(batch, PaddedBatch)
    assert batch.input_ids.tolist() == [
        [2, 5, 3, 0, 0],
        [2, 6, 7, 8, 3],
    ]
    assert batch.attention_mask.tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    assert batch.input_ids.dtype == np.dtype("int64")
    assert batch.attention_mask.dtype == np.dtype("int64")
    assert batch.batch_size == 2
    assert batch.sequence_length == 5

def test_batch_size_splits_sequences() -> None:
    sequences = [
        np.asarray([2, index, 3], dtype=np.int64)
        for index in range(5, 10)
    ]

    batches = list(iter_padded_batches(sequences, batch_size=2))

    assert [batch.batch_size for batch in batches] == [2, 2, 1]
    assert [batch.sequence_length for batch in batches] == [3, 3, 3]

def test_last_batch_can_be_smaller() -> None:
    sequences = [
        np.asarray([2, 5, 3], dtype=np.int64),
        np.asarray([2, 6, 3], dtype=np.int64),
        np.asarray([2, 7, 3], dtype=np.int64),
    ]

    batches = list(iter_padded_batches(sequences, batch_size=2))

    assert batches[-1].input_ids.tolist() == [[2, 7, 3]]
    assert batches[-1].attention_mask.tolist() == [[1, 1, 1]]

def test_batching_is_lazy() -> None:
    consumed: list[int] = []

    def sequences():
        for index in range(4):
            consumed.append(index)
            yield np.asarray([2, index, 3], dtype=np.int64)
    
    batches = iter_padded_batches(sequences(), batch_size=2)

    assert consumed == []

    first_batch = next(batches)

    assert first_batch.batch_size == 2
    assert consumed == [0, 1]

@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(batch_size: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(
            iter_padded_batches(
                [np.asarray([2, 3], dtype=np.int64)],
                batch_size=batch_size,
            )
        )
    
def test_batching_rejects_wrong_dtype() -> None:
    sequences = [
        np.asarray([2, 3], dtype=np.int32),
    ]

    with pytest.raises(ValueError, match="int64"):
        list(iter_padded_batches(sequences, batch_size=1))

def test_batching_rejects_empty_sequence() -> None:
    sequences = [
        np.asarray([], dtype=np.int64),
    ]

    with pytest.raises(ValueError, match="must not be empty"):
        list(iter_padded_batches(sequences, batch_size=1))