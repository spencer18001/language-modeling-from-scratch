from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.experiment import ExperimentLogger
from cs336_basics.nn import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping


def _load_tokens(path: Path, dtype: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    return np.memmap(path, mode="r", dtype=np.dtype(dtype))


@torch.no_grad()
def _estimate_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        losses.append(cross_entropy(logits, y).item())
    model.train()
    return sum(losses) / len(losses)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CS336 Transformer language model.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, default=None)
    parser.add_argument("--data-dtype", type=str, default="uint16")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/debug"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--experiment-notes", type=str, default="")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=5_000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(args.output_dir, args.run_name)
    logger.write_config(vars(args))
    logger.start_run(vars(args), command=" ".join(sys.argv), notes=args.experiment_notes)

    train_data = _load_tokens(args.train_data, args.data_dtype)
    valid_data = _load_tokens(args.valid_data, args.data_dtype) if args.valid_data is not None else None

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(device),
    )
    if args.compile:
        model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        logger.log_event("resume", {"checkpoint": args.resume, "start_iter": start_iter})

    model.train()
    start_time = time.perf_counter()
    for iteration in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        _set_optimizer_lr(optimizer, lr)

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        completed_iter = iteration + 1
        should_eval = completed_iter == 1 or completed_iter % args.eval_interval == 0
        if should_eval:
            train_loss = loss.item()
            valid_loss = None
            if valid_data is not None:
                valid_loss = _estimate_loss(
                    model,
                    valid_data,
                    args.batch_size,
                    args.context_length,
                    device,
                    args.eval_batches,
                )
            elapsed = time.perf_counter() - start_time
            metrics = {
                "iteration": completed_iter,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "lr": lr,
                "elapsed_seconds": elapsed,
                "tokens_processed": completed_iter * args.batch_size * args.context_length,
            }
            print(json.dumps(metrics), flush=True)
            logger.log_metrics(metrics)

        should_checkpoint = (
            completed_iter == args.max_iters
            or (args.checkpoint_interval > 0 and completed_iter % args.checkpoint_interval == 0)
        )
        if should_checkpoint:
            checkpoint_path = checkpoint_dir / f"iter_{completed_iter:06d}.pt"
            save_checkpoint(model, optimizer, completed_iter, checkpoint_path)
            logger.log_event("checkpoint", {"iteration": completed_iter, "path": checkpoint_path})


if __name__ == "__main__":
    main()
