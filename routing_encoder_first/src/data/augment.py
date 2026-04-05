from __future__ import annotations

from typing import Tuple

import torch


def _apply_transform(xy: torch.Tensor, transform_id: int) -> torch.Tensor:
    x = xy[..., [0]]
    y = xy[..., [1]]
    mapping = {
        0: torch.cat((x, y), dim=-1),
        1: torch.cat((1 - x, y), dim=-1),
        2: torch.cat((x, 1 - y), dim=-1),
        3: torch.cat((1 - x, 1 - y), dim=-1),
        4: torch.cat((y, x), dim=-1),
        5: torch.cat((1 - y, x), dim=-1),
        6: torch.cat((y, 1 - x), dim=-1),
        7: torch.cat((1 - y, 1 - x), dim=-1),
    }
    return mapping[transform_id]


def random_symmetry_transform(
    depot_xy: torch.Tensor,
    coords: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    transform_id = int(torch.randint(0, 8, (1,), generator=generator).item())
    return _apply_transform(depot_xy, transform_id), _apply_transform(coords, transform_id)
