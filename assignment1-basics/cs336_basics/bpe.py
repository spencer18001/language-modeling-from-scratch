from __future__ import annotations

from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from os import PathLike

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))


def _initial_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: BYTE_TOKENS[i] for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(token) for token in special_tokens)
    return re.split(pattern, text)


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in _split_on_special_tokens(text, special_tokens):
        for match in re.finditer(PAT, chunk):
            token = tuple(BYTE_TOKENS[b] for b in match.group().encode("utf-8"))
            if token:
                counts[token] += 1
    return counts


def _count_chunk(args: tuple[str, list[str]]) -> Counter[tuple[bytes, ...]]:
    text, special_tokens = args
    return _pretoken_counts(text, special_tokens)


def _pretoken_counts_parallel(
    input_path: str | PathLike,
    special_tokens: list[str],
    num_processes: int | None,
) -> Counter[tuple[bytes, ...]]:
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    if num_processes <= 1:
        with open(input_path, encoding="utf-8") as file:
            return _pretoken_counts(file.read(), special_tokens)

    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    tasks: list[tuple[str, list[str]]] = [] # (chunk_text, special_tokens)
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, split_token)
        for start, end in zip(boundaries, boundaries[1:], strict=False):
            file.seek(start)
            text = file.read(end - start).decode("utf-8", errors="ignore")
            tasks.append((text, special_tokens))

    if len(tasks) <= 1:
        return _count_chunk(tasks[0]) if tasks else Counter()

    counts: Counter[tuple[bytes, ...]] = Counter()
    with Pool(processes=min(num_processes, len(tasks))) as pool:
        for partial_counts in pool.imap_unordered(_count_chunk, tasks):
            counts.update(partial_counts)
    return counts


def _count_pairs(pretoken_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for token, frequency in pretoken_counts.items():
        if len(token) < 2:
            continue
        for left, right in zip(token, token[1:], strict=False):
            pair_counts[(left, right)] += frequency
    return pair_counts


def _pairs_with_multiplicity(token: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    return Counter(zip(token, token[1:], strict=False))


def _build_pair_indexes(
    pretoken_counts: Counter[tuple[bytes, ...]],
) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for token, frequency in pretoken_counts.items():
        for pair, count in _pairs_with_multiplicity(token).items():
            pair_counts[pair] += count * frequency
            pair_to_tokens[pair].add(token)

    return pair_counts, pair_to_tokens


def _merge_pair(token: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    left, right = pair
    merged = left + right
    result: list[bytes] = []
    index = 0
    while index < len(token):
        if index + 1 < len(token) and token[index] == left and token[index + 1] == right:
            result.append(merged)
            index += 2
        else:
            result.append(token[index])
            index += 1
    return tuple(result)


def _remove_token_from_pair_indexes(
    token: tuple[bytes, ...],
    frequency: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    for pair, count in _pairs_with_multiplicity(token).items():
        pair_counts[pair] -= count * frequency
        if pair_counts[pair] <= 0:
            del pair_counts[pair]

        tokens_with_pair = pair_to_tokens.get(pair)
        if tokens_with_pair is not None:
            tokens_with_pair.discard(token)
            if not tokens_with_pair:
                del pair_to_tokens[pair]


def _add_token_to_pair_indexes(
    token: tuple[bytes, ...],
    frequency: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    for pair, count in _pairs_with_multiplicity(token).items():
        pair_counts[pair] += count * frequency
        pair_to_tokens[pair].add(token)


def _train_bpe_from_counts(
    pretoken_counts: Counter[tuple[bytes, ...]],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = _initial_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    pair_counts, pair_to_tokens = _build_pair_indexes(pretoken_counts)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected_tokens = list(pair_to_tokens.get(best_pair, ()))
        merged_counts: Counter[tuple[bytes, ...]] = Counter()

        for token in affected_tokens:
            frequency = pretoken_counts.pop(token, 0)
            if frequency == 0:
                continue

            _remove_token_from_pair_indexes(token, frequency, pair_counts, pair_to_tokens)
            merged_counts[_merge_pair(token, best_pair)] += frequency

        for token, frequency in merged_counts.items():
            pretoken_counts[token] += frequency
            _add_token_to_pair_indexes(token, frequency, pair_counts, pair_to_tokens)

    return vocab, merges


def train_bpe_v1(
    input_path: str | PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Straightforward frequency-table BPE: simple, useful for debugging, slower."""
    vocab = _initial_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    with open(input_path, encoding="utf-8") as file:
        text = file.read()

    pretoken_counts = _pretoken_counts(text, special_tokens)

    while len(vocab) < vocab_size:
        pair_counts = _count_pairs(pretoken_counts)
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        next_counts: Counter[tuple[bytes, ...]] = Counter()
        for token, frequency in pretoken_counts.items():
            next_counts[_merge_pair(token, best_pair)] += frequency
        pretoken_counts = next_counts

    return vocab, merges


def train_bpe_v2(
    input_path: str | PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Incremental pair-index BPE: updates only pre-tokens affected by each merge."""
    with open(input_path, encoding="utf-8") as file:
        text = file.read()
    return _train_bpe_from_counts(_pretoken_counts(text, special_tokens), vocab_size, special_tokens)


def train_bpe_v3(
    input_path: str | PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """v2 merge step with multiprocessing pre-token counting."""
    pretoken_counts = _pretoken_counts_parallel(input_path, special_tokens, num_processes)
    return _train_bpe_from_counts(pretoken_counts, vocab_size, special_tokens)


def train_bpe(
    input_path: str | PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return train_bpe_v2(input_path, vocab_size, special_tokens)
