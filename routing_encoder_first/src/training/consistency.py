from __future__ import annotations

import torch
import torch.nn.functional as F


def augmentation_consistency_loss(edge_scores: torch.Tensor, augmented_edge_scores: torch.Tensor) -> torch.Tensor:
    if edge_scores.shape != augmented_edge_scores.shape:
        raise ValueError("Edge-score tensors must have the same shape for consistency loss")

    diagonal = torch.eye(edge_scores.size(-1), device=edge_scores.device, dtype=torch.bool).unsqueeze(0)
    masked_original = edge_scores.masked_fill(diagonal, 0.0)
    masked_augmented = augmented_edge_scores.masked_fill(diagonal, 0.0)

    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=(-2, -1), keepdim=True)
        std = tensor.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (tensor - mean) / std

    return F.mse_loss(normalize(masked_original), normalize(masked_augmented))
