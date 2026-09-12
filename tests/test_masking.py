import numpy as np
import pytest

from minibert.data.masking import IGNORE_INDEX, MASK_ID, apply_mlm_mask


def test_masking_preserves_shapes_and_does_not_modify_input() -> None:
    input_ids = np.asarray(
        [
            [2, 5, 6, 7, 3, 0],
            [2, 8, 9, 3, 0, 0],
        ],
        dtype=np.int64,
    )
    attention_mask = np.asarray(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    original = input_ids.copy()

    result = apply_mlm_mask(
        input_ids,
        attention_mask,
        vocab_size=20,
        rng=np.random.default_rng(123),
        mask_probability=0.5,
    )

    assert result.input_ids.shape == input_ids.shape
    assert result.labels.shape == input_ids.shape
    assert result.input_ids.dtype == np.dtype("int64")
    assert result.labels.dtype == np.dtype("int64")
    assert np.array_equal(input_ids, original)

def test_masking_never_masks_special_or_padding_positions() -> None:
    input_ids = np.asarray(
        [[2, 5, 6, 7, 3, 0]],
        dtype=np.int64,
    )
    attention_mask = np.asarray(
        [[1, 1, 1, 1, 1, 0]],
        dtype=np.int64,
    )

    result = apply_mlm_mask(
        input_ids,
        attention_mask,
        vocab_size=20,
        rng=np.random.default_rng(1),
        mask_probability=1.0,
    )

    assert result.labels[0, 0] == IGNORE_INDEX
    assert result.labels[0, 4] == IGNORE_INDEX
    assert result.labels[0, 5] == IGNORE_INDEX

    assert result.input_ids[0, 0] == 2
    assert result.input_ids[0, 4] == 3
    assert result.input_ids[0, 5] == 0

def test_masking_selects_at_least_one_token() -> None:
    input_ids = np.asarray(
        [[2, 5, 3]],
        dtype=np.int64,
    )
    attention_mask = np.asarray(
        [[1, 1, 1]],
        dtype=np.int64,
    )

    result = apply_mlm_mask(
        input_ids,
        attention_mask,
        vocab_size=20,
        rng=np.random.default_rng(2),
        mask_probability=0.01,
    )

    assert np.sum(result.labels != IGNORE_INDEX) == 1

def test_masking_is_reprodecible_with_same_seed() -> None:
    input_ids = np.asarray(
        [[2, 5, 6, 7, 8, 3]],
        dtype=np.int64,
    )
    attention_mask = np.ones_like(input_ids)

    first = apply_mlm_mask(
        input_ids,
        attention_mask,
        vocab_size=20,
        rng=np.random.default_rng(10),
    )
    second = apply_mlm_mask(
        input_ids,
        attention_mask,
        vocab_size=20,
        rng=np.random.default_rng(10),
    )

    assert np.array_equal(first.input_ids, second.input_ids)
    assert np.array_equal(first.labels, second.labels)

@pytest.mark.parametrize(
    ("input_ids", "attention_mask"),
    [
        (
            np.asarray([2, 5, 3], dtype=np.int64),
            np.asarray([1, 1, 1], dtype=np.int64),
        ),
        (
            np.asarray([[2, 5, 3]], dtype=np.int32),
            np.asarray([[1, 1, 1]], dtype=np.int64),
        ),
    ],
)
def test_masking_rejects_invalid_input(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        apply_mlm_mask(
            input_ids,
            attention_mask,
            vocab_size=20,
            rng=np.random.default_rng(1),
        )

def test_masking_rejects_invalid_probability() -> None:
    input_ids = np.asarray([[2, 5, 3]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)

    with pytest.raises(ValueError, match="mask_probability"):
        apply_mlm_mask(
            input_ids,
            attention_mask,
            vocab_size=20,
            rng=np.random.default_rng(1),
            mask_probability=0.0,
        )