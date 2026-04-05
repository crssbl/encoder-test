from __future__ import annotations

import torch


def edge_disagreement(successors: torch.Tensor) -> torch.Tensor:
    if successors.dim() != 3:
        raise ValueError(f"Expected successors with shape [B, M, N], got {successors.shape}")
    batch_size, num_candidates, _ = successors.shape
    if num_candidates < 2:
        return torch.zeros((), device=successors.device, dtype=torch.float32)

    disagreements = []
    for first in range(num_candidates):
        for second in range(first + 1, num_candidates):
            overlap = (successors[:, first] == successors[:, second]).float().mean(dim=-1)
            disagreements.append(1.0 - overlap)
    stacked = torch.stack(disagreements, dim=0)
    return stacked.mean()


def diversity_regularizer(successors: torch.Tensor) -> torch.Tensor:
    return -edge_disagreement(successors)
