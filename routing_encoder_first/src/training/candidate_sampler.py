from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, Iterable, List

import numpy as np
import torch

from projectors.giant_tour import project_giant_tour
from projectors.split_dp import no_split_solution, split_dp
from utils.geometry import order_to_successor


def _sample_gumbel_like(shape: torch.Size, generator: torch.Generator) -> torch.Tensor:
    uniform = torch.rand(shape, generator=generator)
    uniform = uniform.clamp_(1e-6, 1.0 - 1e-6)
    return -torch.log(-torch.log(uniform))


def _solve_edge_projection_task(task: tuple) -> tuple:
    (
        batch_idx,
        candidate_idx,
        score_matrix,
        coords,
        depot,
        demands,
        capacity,
        anchor_strategy,
        use_no_split,
    ) = task
    projection = project_giant_tour(
        score_matrix=score_matrix,
        coords=coords,
        depot_xy=depot,
        anchor_strategy=anchor_strategy,
    )
    order = projection["order"]
    if use_no_split:
        split_result = no_split_solution(
            giant_tour=order,
            coords=coords,
            demands=demands,
            depot_xy=depot,
            capacity=capacity,
        )
    else:
        split_result = split_dp(
            giant_tour=order,
            coords=coords,
            demands=demands,
            depot_xy=depot,
            capacity=capacity,
        )
    payload = {
        "order": order,
        "cycle_order": projection["cycle_order"],
        "successor": projection["successor"],
        "routes": split_result["routes"],
        "segments": split_result["segments"],
        "subtours_before_patch": projection["subtours_before_patch"],
    }
    return (
        batch_idx,
        candidate_idx,
        float(split_result["cost"]),
        projection["successor"],
        len(split_result["routes"]),
        bool(split_result["feasible"]),
        True,
        payload,
    )


def _solve_order_projection_task(task: tuple) -> tuple:
    batch_idx, candidate_idx, order, coords, depot, demands, capacity, use_no_split = task
    successor = order_to_successor(order)
    if use_no_split:
        split_result = no_split_solution(
            giant_tour=order,
            coords=coords,
            demands=demands,
            depot_xy=depot,
            capacity=capacity,
        )
    else:
        split_result = split_dp(
            giant_tour=order,
            coords=coords,
            demands=demands,
            depot_xy=depot,
            capacity=capacity,
        )
    payload = {
        "order": order,
        "cycle_order": order,
        "successor": successor,
        "routes": split_result["routes"],
        "segments": split_result["segments"],
        "subtours_before_patch": 1,
    }
    return (
        batch_idx,
        candidate_idx,
        float(split_result["cost"]),
        successor,
        len(split_result["routes"]),
        bool(split_result["feasible"]),
        True,
        payload,
    )


def _run_tasks(
    tasks: list[tuple],
    worker_fn,
    num_workers: int,
    executor: Executor | None = None,
) -> Iterable[tuple]:
    if num_workers <= 1 or len(tasks) <= 1:
        return [worker_fn(task) for task in tasks]
    if executor is not None:
        if isinstance(executor, ProcessPoolExecutor):
            chunksize = max(1, len(tasks) // max(1, num_workers * 4))
            return list(executor.map(worker_fn, tasks, chunksize=chunksize))
        return list(executor.map(worker_fn, tasks))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(worker_fn, tasks))


def build_candidates_from_edge_scores(
    edge_scores: torch.Tensor,
    coords: torch.Tensor,
    depot_xy: torch.Tensor,
    demands: torch.Tensor,
    num_candidates: int,
    noise_scale: float,
    capacity: float,
    anchor_strategy: str,
    generator: torch.Generator,
    include_greedy_candidate: bool = True,
    use_no_split: bool = False,
    num_workers: int = 1,
    executor: Executor | None = None,
) -> Dict[str, object]:
    scores_cpu = edge_scores.detach().float().cpu().numpy()
    coords_cpu = coords.detach().float().cpu().numpy()
    depot_cpu = depot_xy.detach().float().cpu().numpy()
    demands_cpu = demands.detach().float().cpu().numpy()

    batch_size, problem_size, _ = scores_cpu.shape
    costs = torch.empty(batch_size, num_candidates, dtype=torch.float32)
    successors = torch.empty(batch_size, num_candidates, problem_size, dtype=torch.long)
    route_counts = torch.empty(batch_size, num_candidates, dtype=torch.long)
    split_feasible = torch.zeros(batch_size, num_candidates, dtype=torch.bool)
    projector_valid = torch.zeros(batch_size, num_candidates, dtype=torch.bool)
    route_payloads: List[List[Dict[str, object] | None]] = [[None] * num_candidates for _ in range(batch_size)]

    perturbed_scores: list[np.ndarray] = []
    noise_shape = torch.Size([batch_size, problem_size, problem_size])
    for candidate_idx in range(num_candidates):
        if candidate_idx == 0 and include_greedy_candidate:
            perturbed_scores.append(scores_cpu.copy())
        else:
            noise = _sample_gumbel_like(noise_shape, generator).numpy()
            perturbed_scores.append(scores_cpu + noise_scale * noise)

    tasks = []
    for candidate_idx, perturbed in enumerate(perturbed_scores):
        for batch_idx in range(batch_size):
            tasks.append(
                (
                    batch_idx,
                    candidate_idx,
                    perturbed[batch_idx],
                    coords_cpu[batch_idx],
                    depot_cpu[batch_idx, 0],
                    demands_cpu[batch_idx],
                    capacity,
                    anchor_strategy,
                    use_no_split,
                )
            )

    for result in _run_tasks(tasks, _solve_edge_projection_task, num_workers=num_workers, executor=executor):
        batch_idx, candidate_idx, cost, successor, route_count, feasible, valid, payload = result
        costs[batch_idx, candidate_idx] = cost
        successors[batch_idx, candidate_idx] = torch.from_numpy(successor)
        route_counts[batch_idx, candidate_idx] = route_count
        split_feasible[batch_idx, candidate_idx] = feasible
        projector_valid[batch_idx, candidate_idx] = valid
        route_payloads[batch_idx][candidate_idx] = payload

    return {
        "costs": costs,
        "successors": successors,
        "route_counts": route_counts,
        "split_feasible": split_feasible,
        "projector_valid": projector_valid,
        "payloads": route_payloads,
    }


def build_candidates_from_order_logits(
    order_logits: torch.Tensor,
    coords: torch.Tensor,
    depot_xy: torch.Tensor,
    demands: torch.Tensor,
    num_candidates: int,
    noise_scale: float,
    capacity: float,
    generator: torch.Generator,
    include_greedy_candidate: bool = True,
    use_no_split: bool = False,
    num_workers: int = 1,
    executor: Executor | None = None,
) -> Dict[str, object]:
    logits_cpu = order_logits.detach().float().cpu()
    coords_cpu = coords.detach().float().cpu().numpy()
    depot_cpu = depot_xy.detach().float().cpu().numpy()
    demands_cpu = demands.detach().float().cpu().numpy()

    batch_size, problem_size = logits_cpu.shape
    costs = torch.empty(batch_size, num_candidates, dtype=torch.float32)
    successors = torch.empty(batch_size, num_candidates, problem_size, dtype=torch.long)
    route_counts = torch.empty(batch_size, num_candidates, dtype=torch.long)
    split_feasible = torch.zeros(batch_size, num_candidates, dtype=torch.bool)
    projector_valid = torch.ones(batch_size, num_candidates, dtype=torch.bool)
    payloads: List[List[Dict[str, object] | None]] = [[None] * num_candidates for _ in range(batch_size)]

    noisy_logits_list = []
    for candidate_idx in range(num_candidates):
        if candidate_idx == 0 and include_greedy_candidate:
            noisy_logits_list.append(logits_cpu.clone())
        else:
            noisy_logits_list.append(logits_cpu + noise_scale * _sample_gumbel_like(logits_cpu.shape, generator))

    tasks = []
    for candidate_idx, noisy_logits in enumerate(noisy_logits_list):
        order_tensor = noisy_logits.argsort(dim=-1)
        for batch_idx in range(batch_size):
            tasks.append(
                (
                    batch_idx,
                    candidate_idx,
                    order_tensor[batch_idx].tolist(),
                    coords_cpu[batch_idx],
                    depot_cpu[batch_idx, 0],
                    demands_cpu[batch_idx],
                    capacity,
                    use_no_split,
                )
            )

    for result in _run_tasks(tasks, _solve_order_projection_task, num_workers=num_workers, executor=executor):
        batch_idx, candidate_idx, cost, successor, route_count, feasible, valid, payload = result
        costs[batch_idx, candidate_idx] = cost
        successors[batch_idx, candidate_idx] = torch.from_numpy(successor)
        route_counts[batch_idx, candidate_idx] = route_count
        split_feasible[batch_idx, candidate_idx] = feasible
        projector_valid[batch_idx, candidate_idx] = valid
        payloads[batch_idx][candidate_idx] = payload

    return {
        "costs": costs,
        "successors": successors,
        "route_counts": route_counts,
        "split_feasible": split_feasible,
        "projector_valid": projector_valid,
        "payloads": payloads,
    }
