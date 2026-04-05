from __future__ import annotations

from typing import Any, Dict

from eval.compare_hgs import compare_with_hgs
from eval.compare_ortools import compare_with_ortools
from eval.compare_pyvrp import compare_with_pyvrp


def run_baseline(instance: Dict[str, Any], method: str) -> Dict[str, Any]:
    if method == "pyvrp":
        return compare_with_pyvrp(instance)
    if method == "hgs":
        return compare_with_hgs(instance)
    if method == "ortools":
        return compare_with_ortools(instance)
    if method in {"pomo", "polynet", "neurolkh"}:
        return {"status": "skipped", "reason": f"{method} baseline is left for a later integration pass."}
    raise ValueError(f"Unsupported baseline method: {method}")
