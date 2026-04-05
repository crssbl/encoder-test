from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch


@dataclass
class FixedValidationDataset:
    depot: torch.Tensor
    coords: torch.Tensor
    demands: torch.Tensor
    capacities: torch.Tensor
    source_path: str

    def __len__(self) -> int:
        return int(self.depot.size(0))

    def slice(self, indices: torch.Tensor) -> "FixedValidationDataset":
        return FixedValidationDataset(
            depot=self.depot[indices],
            coords=self.coords[indices],
            demands=self.demands[indices],
            capacities=self.capacities[indices],
            source_path=self.source_path,
        )

    def iter_batches(self, batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        total = len(self)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            yield {
                "depot": self.depot[start:end],
                "coords": self.coords[start:end],
                "demands": self.demands[start:end],
                "capacities": self.capacities[start:end],
                "problem_size": int(self.coords.size(1)),
                "distribution": "fixed_validation",
            }


def load_polynet_cvrp_dataset(path: str | Path) -> FixedValidationDataset:
    path = Path(path)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    depots = []
    coords = []
    demands = []
    capacities = []

    for instance in payload:
        depot_xy, customer_xy, customer_demands, capacity = instance
        depots.append(depot_xy)
        coords.append(customer_xy)
        demands.append([float(d) / float(capacity) for d in customer_demands])
        capacities.append(float(capacity))

    return FixedValidationDataset(
        depot=torch.tensor(depots, dtype=torch.float32).unsqueeze(1),
        coords=torch.tensor(coords, dtype=torch.float32),
        demands=torch.tensor(demands, dtype=torch.float32),
        capacities=torch.tensor(capacities, dtype=torch.float32),
        source_path=str(path),
    )


def make_fixed_subset(dataset: FixedValidationDataset, subset_size: int, seed: int) -> FixedValidationDataset:
    if subset_size >= len(dataset):
        return dataset
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    return dataset.slice(indices)
