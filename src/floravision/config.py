"""Experiment configuration: one dataclass, loadable from YAML.

The defaults are the settings of the published model, the same as the original notebook. To
change any of them, pass a YAML file with just the keys to override (``--config``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    seed: int = 42

    # Data
    img_size: int = 224
    batch_size: int = 32
    val_size: float = 0.15  # 70 / 15 / 15 split
    test_size: float = 0.15
    shuffle_buffer: int = 1000

    # Augmentation (training only)
    aug_rotation: float = 0.25
    aug_brightness: float = 0.20
    aug_contrast: float = 0.35
    aug_zoom: float = 0.30
    aug_translation: float = 0.15

    # Model
    pretrained_weights: str | None = "imagenet"
    dense_units: int = 128
    dropout: float = 0.40
    label_smoothing: float = 0.10

    # Phase 1: frozen backbone, train the classification head only
    head_epochs: int = 20
    head_lr: float = 5e-4
    head_lr_factor: float = 0.5
    head_min_lr: float = 1e-6
    head_patience: int = 5
    lr_patience: int = 2

    # Phase 2: fine-tune the last layers of the backbone (BatchNorm stays frozen)
    fine_tune_epochs: int = 10
    fine_tune_lr: float = 5e-6
    fine_tune_layers: int = 15
    fine_tune_patience: int = 3

    def __post_init__(self) -> None:
        if not 0 < self.val_size + self.test_size < 1:
            raise ValueError("val_size + test_size must be between 0 and 1.")
        if self.img_size < 32:
            raise ValueError("img_size must be at least 32 (the smallest ResNet-50 input).")

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, **overrides) -> Config:
        """Load a config file (if given), then apply keyword overrides that are not None."""
        values: dict[str, Any] = {}
        if path is not None:
            with open(path, encoding="utf-8") as f:
                values = yaml.safe_load(f) or {}
        values.update({k: v for k, v in overrides.items() if v is not None})

        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
