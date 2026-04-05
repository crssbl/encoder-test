from __future__ import annotations

import torch
import torch.nn as nn

from utils.geometry import build_pair_features


class EdgeHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 32,
        topk_outgoing: int | None = None,
        sparsify_min_n: int = 200,
    ) -> None:
        super().__init__()
        self.topk_outgoing = topk_outgoing
        self.sparsify_min_n = sparsify_min_n

        pair_feature_dim = 7
        self.src_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.dst_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.depot_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.geom_proj = nn.Sequential(
            nn.Linear(pair_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        depot_embedding: torch.Tensor,
        depot_xy: torch.Tensor,
        coords: torch.Tensor,
        demands: torch.Tensor,
    ) -> torch.Tensor:
        _, problem_size, _ = node_embeddings.shape
        src = self.src_proj(node_embeddings).unsqueeze(2)
        dst = self.dst_proj(node_embeddings).unsqueeze(1)
        depot_bias = self.depot_proj(depot_embedding).unsqueeze(1).unsqueeze(2)
        pair_features = build_pair_features(coords, demands, depot_xy)
        geom = self.geom_proj(pair_features)
        hidden = torch.tanh(src + dst + depot_bias + geom)
        scores = self.out_proj(hidden).squeeze(-1)

        diagonal = torch.eye(problem_size, device=scores.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(diagonal, -1e9)

        if self.topk_outgoing is not None and problem_size >= self.sparsify_min_n:
            topk = min(self.topk_outgoing, problem_size - 1)
            topk_values, topk_indices = scores.topk(topk, dim=-1)
            sparse_scores = torch.full_like(scores, -1e9)
            sparse_scores.scatter_(-1, topk_indices, topk_values)
            sparse_scores = sparse_scores.masked_fill(diagonal, -1e9)
            scores = sparse_scores

        return scores
