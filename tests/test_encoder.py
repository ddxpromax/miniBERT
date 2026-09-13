import torch

from minibert.model.encoder import FeedForward, TransformerEncoderBlock, gelu


def test_gelu_preserves_shape() -> None:
    inputs = torch.randn(2, 5, 12)
    outputs = gelu(inputs)

    assert outputs.shape == inputs.shape

def test_feed_forward_changes_hidden_dimension_internally() -> None:
    feed_forward = FeedForward(
        hidden_size=12,
        intermediate_size=48,
    )

    inputs = torch.randn(2, 5, 12)
    outputs = feed_forward(inputs)

    assert outputs.shape == (2, 5, 12)

def test_encoder_block_output_shape() -> None:
    block = TransformerEncoderBlock(
        hidden_size=12,
        num_heads=3,
        intermediate_size=48,
    )

    hidden_states = torch.randn(2, 5, 12)
    attention_mask = torch.ones(2, 5, dtype=torch.long)

    outputs = block(
        hidden_states,
        attention_mask,
    )

    assert outputs.shape == (2, 5, 12)

def test_encoder_block_zeros_padding_positions() -> None:
    block = TransformerEncoderBlock(
        hidden_size=12,
        num_heads=3,
        intermediate_size=48,
    )

    hidden_states = torch.randn(1, 5, 12)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    outputs = block(
        hidden_states,
        attention_mask,
    )

    torch.testing.assert_close(
        outputs[:, 3:, :],
        torch.zeros(1, 2, 12),
    )

def test_encoder_block_gradients_flow() -> None:
    block = TransformerEncoderBlock(
        hidden_size=12,
        num_heads=3,
        intermediate_size=48,
    )

    hidden_states = torch.randn(2, 5, 12, requires_grad=True)
    attention_mask = torch.ones(2, 5, dtype=torch.long)

    outputs = block(
        hidden_states,
        attention_mask,
    )

    loss = outputs.square().mean()
    loss.backward()

    assert hidden_states.grad is not None

    for parameter in block.parameters():
        assert parameter.grad is not None