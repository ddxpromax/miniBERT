import pytest
import torch

from minibert.model.losses import IGNORE_INDEX, masked_language_model_loss


def test_mlm_loss_returns_scalar() -> None:
    logits = torch.randn(2, 5, 10,)
    labels = torch.tensor(
        [
            [-100, -100, 3, -100, 5],
            [-100, 7, -100, -100, -100],
        ],
        dtype=torch.long,
    )

    loss = masked_language_model_loss(
        logits,
        labels,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

def test_mlm_loss_ignores_unselected_positions() -> None:
    logits = torch.randn(1, 4, 10,)
    labels = torch.tensor(
        [[-100, 3, -100, -100]],
        dtype=torch.long,
    )

    modified_logits = logits.clone()
    modified_logits[0, 0] = torch.randn(10)
    modified_logits[0, 2] = torch.randn(10)
    modified_logits[0, 3] = torch.randn(10)

    first_loss = masked_language_model_loss(
        logits,
        labels,
    )
    second_loss = masked_language_model_loss(
        modified_logits,
        labels,
    )

    torch.testing.assert_close(
        first_loss,
        second_loss,
    )

def test_correct_class_with_highest_logit_has_low_loss() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 10.0],
            ]
        ]
    )

    labels = torch.tensor(
        [[2]],
        dtype=torch.long,
    )

    loss = masked_language_model_loss(
        logits,
        labels,
    )

    assert loss.item() < 0.001

def test_loss_matches_manual_single_position() -> None:
    logits = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
            ]
        ]
    )

    labels = torch.tensor(
        [[1]],
        dtype=torch.long,
    )

    loss = masked_language_model_loss(
        logits,
        labels,
    )

    expected = torch.logsumexp(
        logits[0, 0],
        dim=0,
    ) - logits[0, 0, 1]

    torch.testing.assert_close(
        loss,
        expected,
    )

def test_loss_rejects_no_valid_labels() -> None:
    logits = torch.randn(1, 4, 10)
    labels = torch.full(
        (1, 4),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="at least one label"):
        masked_language_model_loss(
            logits,
            labels,
        )

def test_loss_rejects_invalid_label() -> None:
    logits = torch.randn(1, 4, 10)
    labels = torch.tensor(
        [[-100, 10, -100, -100]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="exceed"):
        masked_language_model_loss(
            logits,
            labels,
        )

def test_loss_gradients_flow() -> None:
    logits = torch.randn(2, 5, 10, requires_grad=True)
    labels = torch.tensor(
        [
            [-100, 3, -100, 5, -100],
            [-100, -100, 7, -100, -100],
        ],
        dtype=torch.long,
    )

    loss = masked_language_model_loss(
        logits,
        labels,
    )
    loss.backward()

    assert logits.grad is not None