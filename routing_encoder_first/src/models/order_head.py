import torch
import torch.nn as nn


class ScalarOrderHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(node_embeddings).squeeze(-1)
