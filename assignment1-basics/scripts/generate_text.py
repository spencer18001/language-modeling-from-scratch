from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_basics.generation import generate_text
from cs336_basics.nn import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def _load_vocab(path: Path) -> dict[int, bytes]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(token_id): bytes(token_bytes) for token_id, token_bytes in payload.items()}


def _load_merges(path: Path) -> list[tuple[bytes, bytes]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(bytes(left), bytes(right)) for left, right in payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained CS336 Transformer LM.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--end-token", type=str, default="<|endoftext|>")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    tokenizer = Tokenizer(
        vocab=_load_vocab(args.vocab),
        merges=_load_merges(args.merges),
        special_tokens=args.special_token,
    )
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

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(model_state)

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        end_token=args.end_token,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
