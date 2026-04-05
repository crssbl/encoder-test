from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def load_cvrplib_instance(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    coords: List[List[float]] = []
    demands: List[float] = []
    depot_index = 1
    capacity = None

    section = None
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("CAPACITY"):
                capacity = float(line.split(":")[-1].strip())
            elif line == "NODE_COORD_SECTION":
                section = "coords"
            elif line == "DEMAND_SECTION":
                section = "demands"
            elif line == "DEPOT_SECTION":
                section = "depot"
            elif line == "EOF":
                break
            elif section == "coords":
                _, x, y = line.split()
                coords.append([float(x), float(y)])
            elif section == "demands":
                _, demand = line.split()
                demands.append(float(demand))
            elif section == "depot":
                if line != "-1":
                    depot_index = int(line)

    if capacity is None:
        raise ValueError(f"CAPACITY missing in {path}")

    coords_np = np.asarray(coords, dtype=np.float32)
    demands_np = np.asarray(demands, dtype=np.float32)
    depot_xy = coords_np[depot_index - 1]

    customer_mask = np.ones(len(coords_np), dtype=bool)
    customer_mask[depot_index - 1] = False
    customer_xy = coords_np[customer_mask]
    customer_demands = demands_np[customer_mask] / capacity

    return {
        "depot": depot_xy[None, :],
        "coords": customer_xy,
        "demands": customer_demands,
        "capacity": np.array([1.0], dtype=np.float32),
    }
