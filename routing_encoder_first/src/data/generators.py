from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


DEMAND_SCALERS = {
    20: 30,
    50: 40,
    100: 50,
    150: 60,
    200: 70,
    300: 80,
    500: 100,
    1000: 125,
}


def demand_scaler_for_size(problem_size: int) -> int:
    if problem_size in DEMAND_SCALERS:
        return DEMAND_SCALERS[problem_size]
    if problem_size < 100:
        return 40
    if problem_size < 300:
        return 70
    if problem_size < 700:
        return 100
    return 125


def _sample_demands(
    batch_size: int,
    problem_size: int,
    generator: torch.Generator,
    mode: str = "standard",
) -> torch.Tensor:
    scaler = float(demand_scaler_for_size(problem_size))
    if mode == "standard":
        raw = torch.randint(1, 10, (batch_size, problem_size), generator=generator)
    elif mode == "skewed":
        raw = torch.ceil(torch.rand(batch_size, problem_size, generator=generator).pow(0.35) * 9.0)
        raw = raw.clamp_min(1.0)
    else:
        raise ValueError(f"Unsupported demand mode: {mode}")
    return raw / scaler


def generate_uniform_batch(batch_size: int, problem_size: int, generator: torch.Generator) -> Dict[str, torch.Tensor]:
    depot_xy = torch.rand(batch_size, 1, 2, generator=generator)
    node_xy = torch.rand(batch_size, problem_size, 2, generator=generator)
    node_demand = _sample_demands(batch_size, problem_size, generator, mode="standard")
    return {"depot": depot_xy, "coords": node_xy, "demands": node_demand}


def generate_clustered_batch(
    batch_size: int,
    problem_size: int,
    generator: torch.Generator,
    num_clusters: int = 5,
    cluster_std: float = 0.08,
) -> Dict[str, torch.Tensor]:
    depot_xy = torch.rand(batch_size, 1, 2, generator=generator)
    centers = torch.rand(batch_size, num_clusters, 2, generator=generator)
    assignments = torch.randint(0, num_clusters, (batch_size, problem_size), generator=generator)
    chosen_centers = centers.gather(1, assignments.unsqueeze(-1).expand(-1, -1, 2))
    coords = chosen_centers + cluster_std * torch.randn(batch_size, problem_size, 2, generator=generator)
    coords = coords.clamp(0.0, 1.0)
    demands = _sample_demands(batch_size, problem_size, generator, mode="skewed")
    return {"depot": depot_xy, "coords": coords, "demands": demands}


def generate_mixed_batch(batch_size: int, problem_size: int, generator: torch.Generator) -> Dict[str, torch.Tensor]:
    selector = torch.rand(batch_size, generator=generator)
    uniform_batch = generate_uniform_batch(batch_size, problem_size, generator)
    clustered_batch = generate_clustered_batch(batch_size, problem_size, generator)

    mixed = {}
    mask = selector < 0.5
    mask_expanded_coords = mask[:, None, None]
    mask_expanded_demands = mask[:, None]
    mixed["depot"] = torch.where(mask_expanded_coords, uniform_batch["depot"], clustered_batch["depot"])
    mixed["coords"] = torch.where(mask_expanded_coords, uniform_batch["coords"], clustered_batch["coords"])
    mixed["demands"] = torch.where(mask_expanded_demands, uniform_batch["demands"], clustered_batch["demands"])
    return mixed


def sample_batch(
    distribution: str,
    batch_size: int,
    problem_size: int,
    generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    if distribution == "uniform":
        batch = generate_uniform_batch(batch_size, problem_size, generator)
    elif distribution == "clustered":
        batch = generate_clustered_batch(batch_size, problem_size, generator)
    elif distribution == "mixed":
        batch = generate_mixed_batch(batch_size, problem_size, generator)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    batch["distribution"] = distribution
    batch["problem_size"] = problem_size
    return batch
