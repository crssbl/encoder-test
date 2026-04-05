from __future__ import annotations

import torch
import torch.nn as nn

from models.edge_head import EdgeHead
from models.order_head import ScalarOrderHead
from utils.geometry import build_node_features


class RoutingTransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        use_polar: bool,
        use_local_density: bool,
        density_k: int,
    ) -> None:
        super().__init__()
        node_feature_dim = 4 + int(use_polar) + int(use_local_density)
        self.use_polar = use_polar
        self.use_local_density = use_local_density
        self.density_k = density_k

        self.depot_proj = nn.Linear(2, embedding_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(node_feature_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.type_embedding = nn.Embedding(2, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embedding_dim))

    def forward(self, depot_xy: torch.Tensor, coords: torch.Tensor, demands: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = build_node_features(
            depot_xy=depot_xy,
            coords=coords,
            demands=demands,
            use_polar=self.use_polar,
            use_local_density=self.use_local_density,
            density_k=self.density_k,
        )
        depot_type = self.type_embedding(torch.zeros(depot_xy.size(0), 1, dtype=torch.long, device=depot_xy.device))
        node_type = self.type_embedding(torch.ones(coords.size(0), coords.size(1), dtype=torch.long, device=coords.device))
        tokens = torch.cat((self.depot_proj(depot_xy) + depot_type, self.node_proj(node_features) + node_type), dim=1)
        encoded = self.encoder(tokens)
        return encoded[:, 1:, :], encoded[:, 0, :]


class EncoderFirstModel(nn.Module):
    def __init__(self, model_cfg: dict, density_k: int) -> None:
        super().__init__()
        self.encoder = RoutingTransformerEncoder(
            embedding_dim=model_cfg["embedding_dim"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            ff_dim=model_cfg["ff_dim"],
            dropout=model_cfg["dropout"],
            use_polar=model_cfg["use_polar"],
            use_local_density=model_cfg["use_local_density"],
            density_k=density_k,
        )
        self.edge_head = EdgeHead(
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["edge_hidden_dim"],
            topk_outgoing=model_cfg["topk_outgoing"],
            sparsify_min_n=model_cfg["sparsify_min_n"],
        )
        self.order_head = ScalarOrderHead(embedding_dim=model_cfg["embedding_dim"])

    def forward(
        self,
        depot_xy: torch.Tensor,
        coords: torch.Tensor,
        demands: torch.Tensor,
        compute_order_logits: bool = True,
    ):
        node_embeddings, depot_embedding = self.encoder(depot_xy, coords, demands)
        edge_scores = self.edge_head(node_embeddings, depot_embedding, depot_xy, coords, demands)
        order_logits = self.order_head(node_embeddings) if compute_order_logits else None
        return edge_scores, node_embeddings, depot_embedding, order_logits
