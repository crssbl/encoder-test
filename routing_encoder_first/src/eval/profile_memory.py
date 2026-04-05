from __future__ import annotations

from typing import Iterable

import torch


def reset_peak_memory(device_ids: Iterable[int]) -> None:
    if not torch.cuda.is_available():
        return
    for device_id in device_ids:
        torch.cuda.reset_peak_memory_stats(device_id)


def get_peak_memory_mb(device_ids: Iterable[int]) -> float:
    if not torch.cuda.is_available():
        return 0.0
    peaks = [torch.cuda.max_memory_allocated(device_id) / (1024 ** 2) for device_id in device_ids]
    return float(max(peaks) if peaks else 0.0)
