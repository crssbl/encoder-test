from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from training.trainer import EncoderFirstTrainer
from utils.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run encoder-first CVRP experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate the latest checkpoint.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_runtime(config: dict) -> None:
    performance_cfg = config.get("performance", {})
    if not torch.cuda.is_available():
        return

    allow_tf32 = bool(performance_cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(performance_cfg.get("cudnn_benchmark", True))

    precision = performance_cfg.get("float32_matmul_precision", "high")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(int(config["seed"]))
    configure_runtime(config)
    trainer = EncoderFirstTrainer(config)
    trainer.run(eval_only=args.eval_only)


if __name__ == "__main__":
    main()
