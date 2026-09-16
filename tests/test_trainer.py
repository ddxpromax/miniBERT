import numpy as np
import torch

from minibert.data.token_storage import TokenizedCorpus, write_tokenized_corpus
from minibert.model.mini_bert import MiniBERT
from minibert.training.trainer import evaluate_epoch, train_epoch
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS, Vocabulary

def build_tiny_corpus(tmp_path):
    vocabulary = Vocabulary(
        [
            *SPECIAL_TOKENS,
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
        ]
    )

    output_dir = tmp_path / "encoded"
    write_tokenized_corpus(
        [
            [5, 6, 7, 8, 9],
            [6, 7, 8, 9],
            [5, 6, 7],
        ],
        output_dir,
        vocabulary,
    )

    return (
        TokenizedCorpus(output_dir),
        vocabulary,
    )

def build_tiny_model(vocab_size: int) -> MiniBERT:
    return MiniBERT(
        vocab_size=vocab_size,
        max_positions=8,
        hidden_size=12,
        num_heads=3,
        intermediate_size=48,
        num_layers=2,
    )

def test_train_epoch_updates_parameters(tmp_path) -> None:
    corpus, vocabulary = build_tiny_corpus(tmp_path)
    model = build_tiny_model(len(vocabulary))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    first_parameter = next(model.parameters())
    before = first_parameter.detach().clone()

    metrics = train_epoch(
        model,
        corpus,
        optimizer,
        max_length=8,
        batch_size=2,
        vocabulary_size=len(vocabulary),
        rng=np.random.default_rng(1),
        device=torch.device("cpu"),
        max_batches=2,
    )
    
    after = first_parameter.detach()

    assert metrics.batches == 2
    assert metrics.masked_positions > 0
    assert np.isfinite(metrics.loss)
    assert not torch.equal(before, after)

def test_evaluate_epoch_does_not_create_gradients(
    tmp_path,
) -> None:
    corpus, vocabulary = build_tiny_corpus(tmp_path)
    model = build_tiny_model(len(vocabulary))

    metrics = evaluate_epoch(
        model,
        corpus,
        max_length=8,
        batch_size=2,
        vocabulary_size=len(vocabulary),
        rng=np.random.default_rng(2),
        device=torch.device("cpu"),
        max_batches=1,
    )

    assert metrics.batches == 1
    assert metrics.masked_positions > 0
    assert np.isfinite(metrics.loss)

    for parameter in model.parameters():
        assert parameter.grad is None