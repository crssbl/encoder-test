from __future__ import annotations

import torch


def median_absolute_deviation(values: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    median = values.median(dim=dim, keepdim=True).values
    mad = (values - median).abs().median(dim=dim).values
    return mad + eps


def build_listwise_targets(costs: torch.Tensor, tau_cost: float = 0.6, eps: float = 1e-6) -> torch.Tensor:
    min_cost = costs.min(dim=-1, keepdim=True).values
    normalized = (costs - min_cost) / median_absolute_deviation(costs, dim=-1, eps=eps).unsqueeze(-1)
    logits = -normalized / tau_cost
    return torch.softmax(logits, dim=-1)


def instance_relative_listwise_loss(
    costs: torch.Tensor,
    energies: torch.Tensor,
    tau_cost: float = 0.6,
    tau_energy: float = 0.6,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    targets = build_listwise_targets(costs, tau_cost=tau_cost, eps=eps)
    log_probs = torch.log_softmax(-energies / tau_energy, dim=-1)
    loss = torch.sum(targets * (torch.log(targets + eps) - log_probs), dim=-1).mean()
    return {"loss": loss, "targets": targets, "log_probs": log_probs}
