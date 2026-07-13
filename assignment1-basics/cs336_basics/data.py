from __future__ import annotations

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length
    if max_start <= 0:
        raise ValueError("dataset must be longer than context_length")

    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([dataset[start : start + context_length] for start in starts])
    y = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])
    return (
        torch.as_tensor(x, dtype=torch.long, device=device),
        torch.as_tensor(y, dtype=torch.long, device=device),
    )
