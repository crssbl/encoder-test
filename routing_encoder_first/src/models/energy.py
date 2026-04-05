from __future__ import annotations

import torch


def score_candidates(edge_scores: torch.Tensor, successors: torch.Tensor) -> torch.Tensor:
    if successors.dim() != 3:
        raise ValueError(f"Expected successors with shape [B, M, N], got {successors.shape}")

    expanded_scores = edge_scores.unsqueeze(1).expand(-1, successors.size(1), -1, -1)
    gathered = expanded_scores.gather(3, successors.unsqueeze(-1)).squeeze(-1)
    return -gathered.mean(dim=-1)
