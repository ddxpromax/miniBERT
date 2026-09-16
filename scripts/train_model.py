"""Train MiniBERT with masked language modeling."""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from pathlib import Path

import numpy as np
import torch

from minibert.data.token_storage import TokenizedCorpus
from minibert.model.mini_bert import MiniBERT
from minibert.training.trainer import evaluate_epoch, train_epoch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None, help="For a small smoke test; omit for a full epoch.")
    parser.add_argument("--device", type=str, default=None, choices=['cpu', 'cuda'])
    parser.add_argument("--metrics", type=Path, default=None, help="Path to the JSONL metrics file.")

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if args.max_length < 3:
        parser.error("--max-length must be at least 3")
    
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    return args

def append_metrics(
    path: Path,
    record: dict[str, object],
) -> None:
    """Append one JSON record without loading previous records."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "a",
        encoding="utf-8",
    ) as metrics_file:
        metrics_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

def choose_device(requested: str | None) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    
    if requested is not None:
        return torch.device(requested)
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(
    output_dir: Path,
    *,
    epoch: int,
    model: MiniBERT,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    validation_loss: float,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "config": vars(args),
    }

    torch.save(
        checkpoint,
        output_dir / f"epoch_{epoch:04d}.pt",
    )

def main() -> None:
    args = parse_args()

    metrics_path = args.metrics

    if metrics_path is None:
        metrics_path = args.output / "metrics.jsonl"

    torch.manual_seed(args.seed)

    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(args.seed + 1)

    device = choose_device(args.device)

    train_corpus = TokenizedCorpus(args.train)
    validation_corpus = TokenizedCorpus(args.validation)

    if train_corpus.vocabulary.tokens != (
        validation_corpus.vocabulary.tokens
    ):
        raise ValueError("train and validation vocabularies do not match")
    
    vocabulary_size = len(train_corpus.vocabulary)

    if args.max_length > 128:
        raise ValueError("this initial training script expects max_length <= 128")
    
    model = MiniBERT(
        vocab_size=vocabulary_size,
        max_positions=args.max_length,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    print(f"Device: {device}")
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Train documents: {len(train_corpus):,}")
    print(f"Validation documents: {len(validation_corpus):,}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        epoch_started_at = perf_counter()

        train_metrics = train_epoch(
            model,
            train_corpus,
            optimizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            vocabulary_size=vocabulary_size,
            rng=rng,
            device=device,
            max_batches=args.max_batches,
        )

        validation_metrics = evaluate_epoch(
            model,
            validation_corpus,
            max_length=args.max_length,
            batch_size=args.batch_size,
            vocabulary_size=vocabulary_size,
            rng=validation_rng,
            device=device,
            max_batches=args.max_batches,
        )

        epoch_seconds = perf_counter() - epoch_started_at

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics.loss:.4f}, "
            f"validation_loss={validation_metrics.loss:.4f}, "
            f"train_batches={train_metrics.batches}, "
            f"validation_batches={validation_metrics.batches}"
            f"seconds={epoch_seconds:.1f}"
        )

        save_checkpoint(
            args.output,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_loss=train_metrics.loss,
            validation_loss=validation_metrics.loss,
            args=args,
        )

        append_metrics(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "validation_loss": validation_metrics.loss,
                "train_batches": train_metrics.batches,
                "validation_batches": validation_metrics.batches,
                "train_masked_positions": train_metrics.masked_positions,
                "validation_masked_positions": validation_metrics.masked_positions,
                "epoch_seconds": epoch_seconds,
                "device": str(device),
                "vocabulary_size": vocabulary_size,
                "max_length": args.max_length,
                "hidden_size": args.hidden_size,
                "num_heads": args.num_heads,
                "intermediate_size": args.intermediate_size,
                "num_layers": args.num_layers,
            },
        )

if __name__ == "__main__":
    main()