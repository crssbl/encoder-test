from __future__ import annotations

import numpy as np

from projectors.cycle_cover import extract_cycles


def _edge_merge_value(
    score_matrix: np.ndarray,
    coords: np.ndarray,
    src: int,
    old_dst: int,
    new_dst: int,
    distance_weight: float,
) -> float:
    old_value = score_matrix[src, old_dst] - distance_weight * np.linalg.norm(coords[src] - coords[old_dst])
    new_value = score_matrix[src, new_dst] - distance_weight * np.linalg.norm(coords[src] - coords[new_dst])
    return float(new_value - old_value)


def patch_subtours(
    successor: np.ndarray,
    score_matrix: np.ndarray,
    coords: np.ndarray,
    distance_weight: float = 0.1,
) -> tuple[np.ndarray, int]:
    patched = successor.copy()
    initial_cycle_count = len(extract_cycles(patched))

    while True:
        cycles = extract_cycles(patched)
        if len(cycles) <= 1:
            break

        best_gain = None
        best_move = None

        for idx_a in range(len(cycles)):
            for idx_b in range(idx_a + 1, len(cycles)):
                cycle_a = cycles[idx_a]
                cycle_b = cycles[idx_b]
                for node_a in cycle_a:
                    succ_a = int(patched[node_a])
                    for node_b in cycle_b:
                        succ_b = int(patched[node_b])
                        gain = _edge_merge_value(score_matrix, coords, node_a, succ_a, succ_b, distance_weight)
                        gain += _edge_merge_value(score_matrix, coords, node_b, succ_b, succ_a, distance_weight)
                        if best_gain is None or gain > best_gain:
                            best_gain = gain
                            best_move = (node_a, succ_a, node_b, succ_b)

        if best_move is None:
            break

        node_a, succ_a, node_b, succ_b = best_move
        patched[node_a] = succ_b
        patched[node_b] = succ_a

    return patched, initial_cycle_count
