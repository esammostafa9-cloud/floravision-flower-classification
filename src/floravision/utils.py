"""Small shared helpers: seeding, logging and JSON I/O."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Fix every source of randomness we control (Python, NumPy, TensorFlow)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import keras

    keras.utils.set_random_seed(seed)


def get_logger(name: str = "floravision") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def save_json(obj, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
