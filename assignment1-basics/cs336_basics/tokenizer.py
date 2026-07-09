from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.special_token_to_id = {
            token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens
        }

        self.special_pattern = None
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile("(" + "|".join(re.escape(token) for token in sorted_tokens) + ")")

    def encode(self, text: str) -> list[int]:
        return list(self._encode_pieces(text))

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self._encode_pieces(text)

    def _encode_pieces(self, text: str) -> Iterator[int]:
        for piece in self._split_on_special_tokens(text):
            if piece in self.special_token_to_id:
                yield self.special_token_to_id[piece]
            else:
                yield from self._encode_ordinary_text(piece)

    def _split_on_special_tokens(self, text: str) -> Iterator[str]:
        if self.special_pattern is None:
            if text:
                yield text
            return

        for piece in self.special_pattern.split(text):
            if piece:
                yield piece

    def _encode_ordinary_text(self, text: str) -> Iterator[int]:
        for match in re.finditer(PAT, text):
            token_bytes = match.group().encode("utf-8")
            token = tuple(BYTE_TOKENS[byte] for byte in token_bytes)
            for bpe_token in self._apply_bpe(token):
                yield self.token_to_id[bpe_token]

    def _apply_bpe(self, token: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if len(token) < 2:
            return token

        token_parts = list(token)
        while True:
            best_rank = None
            best_pair = None
            for pair in zip(token_parts, token_parts[1:], strict=False):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            left, right = best_pair
            merged = left + right
            next_parts: list[bytes] = []
            index = 0
            while index < len(token_parts):
                if (
                    index + 1 < len(token_parts)
                    and token_parts[index] == left
                    and token_parts[index + 1] == right
                ):
                    next_parts.append(merged)
                    index += 2
                else:
                    next_parts.append(token_parts[index])
                    index += 1
            token_parts = next_parts

            if len(token_parts) < 2:
                break

        return tuple(token_parts)
