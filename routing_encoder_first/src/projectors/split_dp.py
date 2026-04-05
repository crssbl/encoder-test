from __future__ import annotations

from typing import Dict, List

import numpy as np


def _prefix_internal_distances(coords: np.ndarray, order: np.ndarray) -> np.ndarray:
    prefix = np.zeros(len(order), dtype=np.float64)
    if len(order) <= 1:
        return prefix
    edge_lengths = np.linalg.norm(coords[order[1:]] - coords[order[:-1]], axis=1)
    prefix[1:] = np.cumsum(edge_lengths)
    return prefix


def _segment_cost(
    order: np.ndarray,
    coords: np.ndarray,
    depot_xy: np.ndarray,
    prefix_edges: np.ndarray,
    start: int,
    end: int,
) -> float:
    first_node = order[start]
    last_node = order[end]
    internal = prefix_edges[end] - prefix_edges[start]
    outbound = np.linalg.norm(coords[first_node] - depot_xy)
    inbound = np.linalg.norm(coords[last_node] - depot_xy)
    return float(outbound + internal + inbound)


def split_dp(
    giant_tour: List[int] | np.ndarray,
    coords: np.ndarray,
    demands: np.ndarray,
    depot_xy: np.ndarray,
    capacity: float = 1.0,
) -> Dict[str, object]:
    order = np.asarray(giant_tour, dtype=np.int64)
    ordered_demands = demands[order]
    if np.any(ordered_demands > capacity + 1e-9):
        return {"routes": [], "segments": [], "cost": float("inf"), "feasible": False}

    n = len(order)
    prefix_demand = np.concatenate(([0.0], np.cumsum(ordered_demands)))
    prefix_edges = _prefix_internal_distances(coords, order)

    dp = np.full(n + 1, np.inf, dtype=np.float64)
    parent = np.full(n + 1, -1, dtype=np.int64)
    dp[0] = 0.0

    for end in range(1, n + 1):
        for start in range(end - 1, -1, -1):
            demand_sum = prefix_demand[end] - prefix_demand[start]
            if demand_sum > capacity + 1e-9:
                break
            route_cost = _segment_cost(order, coords, depot_xy, prefix_edges, start, end - 1)
            candidate = dp[start] + route_cost
            if candidate < dp[end]:
                dp[end] = candidate
                parent[end] = start

    if not np.isfinite(dp[n]):
        return {"routes": [], "segments": [], "cost": float("inf"), "feasible": False}

    segments = []
    pointer = n
    while pointer > 0:
        start = int(parent[pointer])
        segments.append((start, pointer))
        pointer = start
    segments.reverse()

    routes = [order[start:end].tolist() for start, end in segments]
    return {"routes": routes, "segments": segments, "cost": float(dp[n]), "feasible": True}


def no_split_solution(
    giant_tour: List[int] | np.ndarray,
    coords: np.ndarray,
    demands: np.ndarray,
    depot_xy: np.ndarray,
    capacity: float = 1.0,
) -> Dict[str, object]:
    order = np.asarray(giant_tour, dtype=np.int64)
    if demands[order].sum() > capacity + 1e-9:
        return {"routes": [order.tolist()], "segments": [(0, len(order))], "cost": float("inf"), "feasible": False}

    routes = [order.tolist()]
    route_cost = 0.0
    if len(order) > 0:
        route_cost += np.linalg.norm(coords[order[0]] - depot_xy)
        if len(order) > 1:
            route_cost += np.linalg.norm(coords[order[1:]] - coords[order[:-1]], axis=1).sum()
        route_cost += np.linalg.norm(coords[order[-1]] - depot_xy)
    return {"routes": routes, "segments": [(0, len(order))], "cost": float(route_cost), "feasible": True}
