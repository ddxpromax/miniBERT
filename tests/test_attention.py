import pytest
import torch

from minibert.model.attention import MultiHeadSelfAttention


def test_attention_output_shape() -> None:
    attention = MultiHeadSelfAttention(
        hidden_size=12,
        num_heads=3,
    )

    hidden_states = torch.randn(2, 5, 12)
    attention_mask = torch.ones(2, 5, dtype=torch.long)

    outputs = attention(hidden_states, attention_mask)

    assert outputs.shape == (2, 5, 12)
    assert outputs.dtype == torch.float32

def test_attention_zeros_padding_query_positions() -> None:
    attention = MultiHeadSelfAttention(
        hidden_size=12,
        num_heads=3,
    )

    hidden_states = torch.randn(1, 5, 12)
    attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0]],
        dtype=torch.long,
    )

    outputs = attention(
        hidden_states,
        attention_mask,
    )

    torch.testing.assert_close(
        outputs[:, 3:, :],
        torch.zeros(1, 2, 12),
    )

def test_attention_supports_one_token_sequence() -> None:
    attention = MultiHeadSelfAttention(
        hidden_size=12,
        num_heads=3,
    )

    hidden_states = torch.randn(2, 1, 12)
    attention_mask = torch.ones(2, 1, dtype=torch.long)
    
    outputs = attention(
        hidden_states,
        attention_mask,
    )

    assert outputs.shape == (2, 1, 12)

@pytest.mark.parametrize(
    ("hidden_size", "num_heads"),
    [
        (10, 3),
        (12, 0),
        (0, 3),
    ],
)
def test_attention_rejects_invalid_configuration(
    hidden_size: int,
    num_heads: int,
) -> None:
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

def test_attention_rejects_wrong_mask_shape() -> None:
    attention = MultiHeadSelfAttention(
        hidden_size=12,
        num_heads=3,
    )

    hidden_states = torch.randn(2, 5, 12)
    attention_mask = torch.ones(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="shape"):
        attention(
            hidden_states,
            attention_mask,
        )

def test_attention_rejects_wrong_mask_dtype() -> None:
    attention = MultiHeadSelfAttention(
        hidden_size=12,
        num_heads=3,
    )

    hidden_states = torch.randn(2, 5, 12)
    attention_mask = torch.ones(2, 5, dtype=torch.float32)

    with pytest.raises(TypeError, match="torch.long"):
        attention(
            hidden_states,
            attention_mask,
        )

def test_attention_gradients_flow() -> None:
    attention = MultiHeadSelfAttention(
        hidden_size=12,
        num_heads=3,
    )

    hidden_states = torch.randn(2, 5, 12, requires_grad=True)
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    
    outputs = attention(
        hidden_states,
        attention_mask,
    )

    loss = outputs.square().mean()
    loss.backward()

    assert hidden_states.grad is not None

    for parameter in attention.parameters():
        assert parameter.grad is not None