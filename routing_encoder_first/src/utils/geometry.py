from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np
import torch


def euclidean_distance_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_local_density(coords: torch.Tensor, k: int = 8, eps: float = 1e-6) -> torch.Tensor:
    batch_size, problem_size, _ = coords.shape
    if problem_size <= 1:
        return torch.ones(batch_size, problem_size, device=coords.device, dtype=coords.dtype)
    pairwise = torch.cdist(coords, coords)
    diagonal = torch.eye(problem_size, device=coords.device, dtype=torch.bool).unsqueeze(0)
    pairwise = pairwise.masked_fill(diagonal, float("inf"))
    k_eff = min(k, problem_size - 1)
    nearest = pairwise.topk(k_eff, largest=False, dim=-1).values
    mean_distance = nearest.mean(dim=-1)
    return 1.0 / (mean_distance + eps)


def build_node_features(
    depot_xy: torch.Tensor,
    coords: torch.Tensor,
    demands: torch.Tensor,
    use_polar: bool = True,
    use_local_density: bool = True,
    density_k: int = 8,
) -> torch.Tensor:
    depot = depot_xy.expand(-1, coords.size(1), -1)
    relative = coords - depot
    dist_to_depot = torch.linalg.norm(relative, dim=-1, keepdim=True)

    features = [coords, demands.unsqueeze(-1), dist_to_depot]

    if use_polar:
        polar = torch.atan2(relative[..., 1], relative[..., 0]).unsqueeze(-1) / math.pi
        features.append(polar)

    if use_local_density:
        density = compute_local_density(coords, k=density_k).unsqueeze(-1)
        features.append(density)

    return torch.cat(features, dim=-1)


def build_pair_features(coords: torch.Tensor, demands: torch.Tensor, depot_xy: torch.Tensor) -> torch.Tensor:
    problem_size = coords.size(1)
    src = coords[:, :, None, :]
    dst = coords[:, None, :, :]
    delta = dst - src
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True)

    src_demand = demands[:, :, None, None].expand(-1, -1, problem_size, -1)
    dst_demand = demands[:, None, :, None].expand(-1, problem_size, -1, -1)

    depot = depot_xy.expand(-1, coords.size(1), -1)
    depot_dist = torch.linalg.norm(coords - depot, dim=-1, keepdim=True)
    src_depot = depot_dist[:, :, None, :].expand(-1, -1, problem_size, -1)
    dst_depot = depot_dist[:, None, :, :].expand(-1, problem_size, -1, -1)

    return torch.cat((distance, delta, src_demand, dst_demand, src_depot, dst_depot), dim=-1)


def order_to_successor(order: Iterable[int]) -> np.ndarray:
    order_list = list(order)
    successor = np.empty(len(order_list), dtype=np.int64)
    for idx, node in enumerate(order_list):
        successor[node] = order_list[(idx + 1) % len(order_list)]
    return successor


def rotate_order(order: List[int], start_node: int) -> List[int]:
    start_index = order.index(start_node)
    return order[start_index:] + order[:start_index]


def routes_cost_np(routes: List[List[int]], coords: np.ndarray, depot_xy: np.ndarray) -> float:
    total_cost = 0.0
    for route in routes:
        if not route:
            continue
        total_cost += euclidean_distance_np(depot_xy, coords[route[0]])
        for idx in range(len(route) - 1):
            total_cost += euclidean_distance_np(coords[route[idx]], coords[route[idx + 1]])
        total_cost += euclidean_distance_np(coords[route[-1]], depot_xy)
    return total_cost
