from __future__ import annotations

import torch

from cs336_basics.nn import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_p == 1:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    keep_sorted = cumulative_probs <= top_p
    keep_sorted[..., 0] = True
    keep_sorted[..., 1:] = keep_sorted[..., 1:] | (cumulative_probs[..., :-1] < top_p)

    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs * keep_sorted)
    return filtered / filtered.sum(dim=-1, keepdim=True)


@torch.no_grad()
def generate_ids(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    end_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | torch.device | None = None,
) -> list[int]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    model_device = device if device is not None else next(model.parameters()).device
    generated = list(prompt_ids)
    model.eval()

    for _ in range(max_new_tokens):
        context = generated[-model.context_length :]
        input_ids = torch.tensor([context], dtype=torch.long, device=model_device)
        logits = model(input_ids)[0, -1, :] / temperature
        probs = softmax(logits, dim=-1)
        probs = top_p_filter(probs, top_p)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        if end_token_id is not None and next_id == end_token_id:
            break

    return generated


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    end_token: str | None = "<|endoftext|>",
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | torch.device | None = None,
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    end_token_id = None
    if end_token is not None:
        end_token_id = tokenizer.special_token_to_id.get(end_token)
    generated_ids = generate_ids(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        end_token_id=end_token_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )
    return tokenizer.decode(generated_ids)
