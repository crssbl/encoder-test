import json
import logging
import os
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def setup_logger(output_dir: str | os.PathLike[str], name: str = "routing_encoder_first") -> logging.Logger:
    ensure_dir(output_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(Path(output_dir) / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def save_json(payload: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_markdown(text: str, path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
