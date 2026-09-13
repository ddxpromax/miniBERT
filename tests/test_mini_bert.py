import pytest
import torch

from minibert.model.mini_bert import MiniBERT


def build_model() -> MiniBERT:
    return MiniBERT(
        vocab_size=32,
        max_positions=16,
        hidden_size=12,
        num_heads=3,
        intermediate_size=48,
        num_layers=2,
    )

def test_model_output_shape() -> None:
    model = build_model()
    
    input_ids = torch.tensor(
        [
            [2, 5, 6, 7, 3],
            [2, 8, 9, 3, 0],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    logits = model(input_ids, attention_mask)

    assert logits.shape == (2, 5, 32)
    assert logits.dtype == torch.float32

def test_model_has_expected_number_of_layers() -> None:
    model = build_model()

    assert len(model.encoder_layers) == 2

def test_model_rejects_too_long_sequence() -> None:
    model = build_model()

    input_ids = torch.ones(1, 17, dtype=torch.long)

    attention_mask = torch.ones_like(input_ids)

    with pytest.raises(ValueError, match="max_positions"):
        model(input_ids, attention_mask)

def test_model_rejects_wrong_attention_mask_shape() -> None:
    model = build_model()

    input_ids = torch.ones(2, 5, dtype=torch.long)
    attention_mask = torch.ones(2, 4, dtype=torch.long)

    with pytest.raises(ValueError, match="shape"):
        model(input_ids, attention_mask)

def test_model_gradients_flow() -> None:
    model = build_model()

    input_ids = torch.tensor(
        [[2, 5, 6, 7, 3]],
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(input_ids)

    logits = model(input_ids, attention_mask)

    loss = logits.square().mean()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None