"""Shared fixtures: a tiny synthetic flower folder so tests never need the real data."""

import numpy as np
import pytest
from PIL import Image

from floravision.config import Config

CLASSES = ["daisy", "roses", "tulips"]
PER_CLASS = 10


@pytest.fixture
def tiny_dataset(tmp_path):
    """data/<class>/*.jpg, plus a non-image file that must be ignored."""
    root = tmp_path / "flowers"
    rng = np.random.default_rng(0)
    for c, name in enumerate(CLASSES):
        folder = root / name
        folder.mkdir(parents=True)
        for i in range(PER_CLASS):
            # Each class has a distinct mean colour so a model can actually learn something.
            pixels = rng.integers(0, 60, size=(40, 48, 3)) + c * 90
            Image.fromarray(pixels.astype(np.uint8)).save(folder / f"{name}_{i:03d}.jpg")
        (folder / "notes.txt").write_text("not an image")
    return root


@pytest.fixture
def tiny_config():
    return Config(
        img_size=32,  # smallest input ResNet-50 accepts
        batch_size=8,
        val_size=0.2,
        test_size=0.2,
        pretrained_weights=None,  # no download in tests
        dense_units=8,
        head_epochs=1,
        fine_tune_epochs=1,
        fine_tune_layers=5,
    )
