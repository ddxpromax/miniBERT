import numpy as np
import torch

from minibert.data.batching import iter_padded_batches
from minibert.data.masking import apply_mlm_mask
from minibert.data.sequences import iter_training_sequences
from minibert.model.losses import masked_language_model_loss
from minibert.model.mini_bert import MiniBERT
from minibert.data.token_storage import TokenizedCorpus, write_tokenized_corpus
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS, Vocabulary


def test_one_mlm_training_step(tmp_path) -> None:
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

    encoded_dir = tmp_path / "encoded"

    write_tokenized_corpus(
        [
            [5, 6, 7, 8, 9],
            [6, 7, 8],
        ],
        encoded_dir,
        vocabulary,
    )

    corpus = TokenizedCorpus(encoded_dir)

    sequences = iter_training_sequences(
        corpus,
        max_length=8,
    )

    batch = next(
        iter_padded_batches(
            sequences,
            batch_size=2,
        )
    )

    masked = apply_mlm_mask(
        batch.input_ids,
        batch.attention_mask,
        vocab_size=len(vocabulary),
        rng=np.random.default_rng(123),
    )

    model = MiniBERT(
        vocab_size=len(vocabulary),
        max_positions=8,
        hidden_size=12,
        num_heads=3,
        intermediate_size=48,
        num_layers=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    input_ids = torch.from_numpy(
        masked.input_ids,
    )

    attention_mask = torch.from_numpy(
        batch.attention_mask,
    )

    labels = torch.from_numpy(
        masked.labels,
    )

    optimizer.zero_grad()

    logits = model(input_ids, attention_mask)

    loss = masked_language_model_loss(logits, labels)

    assert logits.shape == (2, 7, len(vocabulary))
    assert torch.isfinite(loss)

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert gradients
    assert all(
        gradient is not None
        for gradient in gradients
    )

    original_parameter = next(
        model.parameters()
    ).detach().clone()

    optimizer.step()

    updated_parameter = next(
        model.parameters()
    ).detach()

    assert not torch.equal(
        original_parameter,
        updated_parameter,
    )