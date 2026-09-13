import pytest
import torch

from minibert.model.embeddings import BertEmbeddings, PositionEmbedding


def test_position_embedding_shape() -> None:
    embedding = PositionEmbedding(
        max_positions=16,
        hidden_size=12,
    )

    position_ids = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=torch.long,
    )

    outputs = embedding(position_ids)
    
    assert outputs.shape == (2, 3, 12)

def test_bert_embedding_shape() -> None:
    embeddings = BertEmbeddings(
        vocab_size=32,
        max_positions=16,
        hidden_size=12,
    )

    input_ids = torch.tensor(
        [
            [2, 5, 6, 3],
            [2, 7, 3, 0],
        ],
        dtype=torch.long,
    )

    outputs = embeddings(input_ids)

    assert outputs.shape == (2, 4, 12)

def test_bert_embedding_reuses_token_embedding() -> None:
    embeddings = BertEmbeddings(
        vocab_size=32,
        max_positions=16,
        hidden_size=12,
    )

    assert isinstance(
        embeddings.token_embedding.weight,
        torch.nn.Parameter,
    )

    assert embeddings.token_embedding.weight.shape == (
        32,
        12,
    )

def test_position_embedding_rejects_too_large_position() -> None:
    embedding = PositionEmbedding(
        max_positions=4,
        hidden_size=12,
    )

    position_ids = torch.tensor(
        [[0, 1, 4]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="max_positions"):
        embedding(position_ids)

def test_bert_embedding_gradients_flow() -> None:
    embeddings = BertEmbeddings(
        vocab_size=32,
        max_positions=16,
        hidden_size=12,
    )

    input_ids = torch.tensor(
        [[2, 5, 6, 3]],
        dtype=torch.long,
    )

    outputs = embeddings(input_ids)
    loss = outputs.square().mean()
    loss.backward()

    assert embeddings.token_embedding.weight.grad is not None
    assert embeddings.position_embedding.weight.grad is not None