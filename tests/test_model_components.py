import pytest
import torch

from minibert.model.components import LayerNorm, Linear, TokenEmbedding


def test_token_embedding_shape() -> None:
    embedding = TokenEmbedding(
        vocab_size=20,
        hidden_size=8,
    )

    input_ids = torch.tensor(
        [
            [2, 5, 4],
            [7, 8, 3],
        ],
        dtype=torch.long,
    )

    outputs = embedding(input_ids)

    assert outputs.shape == (2, 3, 8)
    assert outputs.dtype == torch.float32

def test_token_embedding_reuses_same_vector() -> None:
    embedding = TokenEmbedding(
        vocab_size=20,
        hidden_size=8,
    )

    input_ids = torch.tensor(
        [[5, 5]],
        dtype=torch.long,
    )

    outputs = embedding(input_ids)

    torch.testing.assert_close(
        outputs[:, 0, :],
        outputs[:, 1, :],
    )

@pytest.mark.parametrize(
    "input_ids",
    [
        torch.tensor([[1.0]]),
        torch.tensor([[-1]], dtype=torch.long),
        torch.tensor([[20]], dtype=torch.long),
    ],
)
def test_token_embedding_rejects_invalid_ids(
    input_ids: torch.Tensor,
) -> None:
    embedding = TokenEmbedding(
        vocab_size=20,
        hidden_size=8,
    )
    
    with pytest.raises((TypeError, ValueError)):
        embedding(input_ids)

def test_linear_shape() -> None:
    linear = Linear(
        input_size=8,
        output_size=12,
    )

    inputs = torch.randn(2, 5, 8)
    outputs = linear(inputs)

    assert outputs.shape == (2, 5, 12)

def test_linear_supports_no_bias() -> None:
    linear = Linear(
        input_size=8,
        output_size=12,
        bias=False,
    )

    assert linear.bias is None

    inputs = torch.randn(2, 5, 8)
    outputs = linear(inputs)

    assert outputs.shape == (2, 5, 12)

def test_linear_rejects_wrong_input_size() -> None:
    linear = Linear(
        input_size=8,
        output_size=12,
    )

    with pytest.raises(ValueError, match="input_size"):
        linear(torch.randn(2, 5, 7))

def test_layer_norm_shape_and_statistics() -> None:
    layer_norm = LayerNorm(hidden_size=8)

    inputs = torch.randn(2, 5, 8)
    outputs = layer_norm(inputs)

    assert outputs.shape == inputs.shape

    means = outputs.mean(dim=-1)
    variances = outputs.var(dim=-1, unbiased=False)

    torch.testing.assert_close(
        means,
        torch.zeros_like(means),
        atol=1e-5,
        rtol=1e-5,
    )

    torch.testing.assert_close(
        variances,
        torch.ones_like(variances),
        atol=1e-4,
        rtol=1e-4,
    )

def test_components_have_trainable_parameters() -> None:
    modules = [
        TokenEmbedding(20, 8),
        Linear(8, 12),
        LayerNorm(8),
    ]

    for module in modules:
        parameters = list(module.parameters())

        assert parameters
        assert all(parameter.requires_grad for parameter in parameters)

def test_gradients_flow_through_components() -> None:
    embedding = TokenEmbedding(20, 8)
    linear = Linear(8, 4)
    layer_norm = LayerNorm(4)

    input_ids = torch.tensor(
        [[2, 5, 7]],
        dtype=torch.long,
    )

    outputs = embedding(input_ids)
    outputs = linear(outputs)
    outputs = layer_norm(outputs)

    loss = outputs.square().mean()
    loss.backward()

    assert embedding.weight.grad is not None
    assert linear.weight.grad is not None
    assert layer_norm.weight.grad is not None