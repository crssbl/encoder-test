from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment


def project_cycle_cover(score_matrix: np.ndarray) -> np.ndarray:
    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    successor = np.empty(score_matrix.shape[0], dtype=np.int64)
    successor[row_ind] = col_ind
    return successor


def extract_cycles(successor: np.ndarray) -> List[List[int]]:
    visited = np.zeros(len(successor), dtype=bool)
    cycles: List[List[int]] = []
    for start in range(len(successor)):
        if visited[start]:
            continue
        cycle = []
        current = start
        while not visited[current]:
            visited[current] = True
            cycle.append(int(current))
            current = int(successor[current])
        cycles.append(cycle)
    return cycles


def is_valid_single_cycle(successor: np.ndarray) -> bool:
    cycles = extract_cycles(successor)
    return len(cycles) == 1 and len(cycles[0]) == len(successor)
