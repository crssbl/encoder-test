from __future__ import annotations

from typing import Dict, List

import numpy as np

from projectors.cycle_cover import is_valid_single_cycle, project_cycle_cover
from projectors.subtour_patch import patch_subtours
from utils.geometry import rotate_order


def successor_to_order(successor: np.ndarray, start_node: int = 0) -> List[int]:
    order = [int(start_node)]
    current = int(successor[start_node])
    while current != start_node:
        order.append(int(current))
        current = int(successor[current])
    return order


def choose_anchor(order: List[int], coords: np.ndarray, depot_xy: np.ndarray, strategy: str = "nearest_depot") -> int:
    if strategy == "nearest_depot":
        distances = [np.linalg.norm(coords[node] - depot_xy) for node in order]
        return order[int(np.argmin(distances))]
    return order[0]


def project_giant_tour(
    score_matrix: np.ndarray,
    coords: np.ndarray,
    depot_xy: np.ndarray,
    anchor_strategy: str = "nearest_depot",
) -> Dict[str, object]:
    successor = project_cycle_cover(score_matrix)
    successor, subtours_before_patch = patch_subtours(successor, score_matrix, coords)
    if not is_valid_single_cycle(successor):
        raise RuntimeError("Subtour patching failed to produce a single cycle")
    cycle_order = successor_to_order(successor, start_node=0)
    anchor = choose_anchor(cycle_order, coords, depot_xy, strategy=anchor_strategy)
    order = rotate_order(cycle_order, anchor)
    return {
        "successor": successor,
        "cycle_order": cycle_order,
        "order": order,
        "subtours_before_patch": subtours_before_patch,
    }
