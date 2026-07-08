from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_basics.bpe import train_bpe_v3


def _bytes_to_ints(token: bytes) -> list[int]:
    return list(token)


def _serialize_vocab(vocab: dict[int, bytes], path: Path) -> None:
    payload = {str(index): _bytes_to_ints(token) for index, token in vocab.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _serialize_merges(merges: list[tuple[bytes, bytes]], path: Path) -> None:
    payload = [[_bytes_to_ints(left), _bytes_to_ints(right)] for left, right in merges]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizers/tinystories_bpe_10k"))
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--num-processes", type=int, default=None)
    args = parser.parse_args()

    process = psutil.Process(os.getpid())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    start_memory = process.memory_info().rss
    vocab, merges = train_bpe_v3(
        args.input,
        args.vocab_size,
        args.special_token,
        num_processes=args.num_processes,
    )
    elapsed_seconds = time.perf_counter() - start_time
    end_memory = process.memory_info().rss

    longest_id, longest_token = max(vocab.items(), key=lambda item: len(item[1]))
    stats = {
        "input": str(args.input),
        "vocab_size": args.vocab_size,
        "special_tokens": args.special_token,
        "num_merges": len(merges),
        "elapsed_seconds": elapsed_seconds,
        "rss_start_mb": start_memory / 1024 / 1024,
        "rss_end_mb": end_memory / 1024 / 1024,
        "longest_token_id": longest_id,
        "longest_token_num_bytes": len(longest_token),
        "longest_token_text": longest_token.decode("utf-8", errors="replace"),
        "longest_token_bytes": _bytes_to_ints(longest_token),
    }

    _serialize_vocab(vocab, args.output_dir / "vocab.json")
    _serialize_merges(merges, args.output_dir / "merges.json")
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
