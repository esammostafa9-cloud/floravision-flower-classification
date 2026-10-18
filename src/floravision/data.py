"""Dataset loading, splitting and the tf.data input pipeline.

Two data sources give the same output, so the rest of the code does not care which is used:

* **tf_flowers** (default) from TensorFlow Datasets, split by slicing the single ``train``
  split exactly like the original notebook: ``train[:70%]``, ``train[70%:85%]``,
  ``train[85%:]``. The published model was trained and evaluated on these splits.
* **A folder of class sub-folders** (``<root>/<class>/*.jpg``), split with a stratified,
  seeded split. Useful for your own data, and used by the tests so they run offline.

Each split is a ``tf.data.Dataset`` of ``(uint8 image [H, W, 3], int64 label)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import Config

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "val", "test")
TFDS_NAME = "tf_flowers"


@dataclass
class Splits:
    """The train / val / test datasets and the class names their integer labels refer to."""

    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: list[str]
    sizes: dict[str, int]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def __getitem__(self, split: str) -> tf.data.Dataset:
        if split not in SPLITS:
            raise KeyError(split)
        return getattr(self, split)


def align_labels(ds: tf.data.Dataset, data_names: list[str], model_names: list[str]):
    """Re-map a dataset's integer labels into the model's class order.

    A model's output index ``i`` means ``model_names[i]``. Data whose classes are listed in a
    different order (e.g. tf_flowers vs. sorted folder names) would otherwise be scored, or
    labelled, against the wrong class. Fails if the two lists do not hold the same classes.
    """
    if sorted(data_names) != sorted(model_names) or len(set(model_names)) != len(model_names):
        raise ValueError(
            f"The data's classes do not match the model's.\n"
            f"  model: {list(model_names)}\n  data:  {list(data_names)}"
        )
    if list(data_names) == list(model_names):
        return ds
    mapping = tf.constant([model_names.index(name) for name in data_names], dtype=tf.int64)
    return ds.map(lambda image, label: (image, tf.gather(mapping, tf.cast(label, tf.int64))))


# --- tf_flowers (TensorFlow Datasets) ---------------------------------------------------


def tfds_split_specs(val_size: float, test_size: float) -> list[str]:
    """TFDS slicing strings for train / val / test, e.g. ``["train[:70%]", ...]``."""
    train_end = round((1 - val_size - test_size) * 100)
    val_end = round((1 - test_size) * 100)
    if not 0 < train_end < val_end < 100:
        raise ValueError(f"Invalid split sizes: val={val_size}, test={test_size}")
    return [f"train[:{train_end}%]", f"train[{train_end}%:{val_end}%]", f"train[{val_end}%:]"]


def load_tfds_splits(cfg: Config, data_dir: str | Path | None = None) -> Splits:
    """Download (first run only) and split tf_flowers. ``data_dir`` is the TFDS cache folder."""
    import tensorflow_datasets as tfds

    specs = tfds_split_specs(cfg.val_size, cfg.test_size)
    datasets, info = tfds.load(
        TFDS_NAME,
        split=specs,
        as_supervised=True,
        with_info=True,
        data_dir=str(data_dir) if data_dir else None,
    )
    sizes = {name: int(ds.cardinality()) for name, ds in zip(SPLITS, datasets, strict=True)}
    return Splits(*datasets, class_names=list(info.features["label"].names), sizes=sizes)


# --- Folder of class sub-folders --------------------------------------------------------


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)


def list_image_folder(root: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """All image paths under ``root/<class>/``, with labels in sorted class-name order."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    class_names = sorted(
        d.name for d in root.iterdir() if d.is_dir() and any(map(_is_image, os.listdir(d)))
    )
    if len(class_names) < 2:
        raise ValueError(f"Expected at least 2 class folders with images in {root}")

    paths, labels = [], []
    for label, name in enumerate(class_names):
        for fname in sorted(os.listdir(root / name)):
            if _is_image(fname):
                paths.append(str(root / name / fname))
                labels.append(label)
    return np.array(paths), np.array(labels, dtype=np.int64), class_names


def _decode_file(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return image, label


def load_folder_splits(root: str | Path, cfg: Config) -> Splits:
    """Stratified, seeded train / val / test split of an image folder."""
    paths, labels, class_names = list_image_folder(root)
    holdout = cfg.val_size + cfg.test_size
    train_p, rest_p, train_y, rest_y = train_test_split(
        paths, labels, test_size=holdout, stratify=labels, random_state=cfg.seed
    )
    val_p, test_p, val_y, test_y = train_test_split(
        rest_p, rest_y, test_size=cfg.test_size / holdout, stratify=rest_y, random_state=cfg.seed
    )

    def to_ds(p: np.ndarray, y: np.ndarray) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((p, y)).map(_decode_file)

    parts = {"train": (train_p, train_y), "val": (val_p, val_y), "test": (test_p, test_y)}
    return Splits(
        **{name: to_ds(*parts[name]) for name in SPLITS},
        class_names=class_names,
        sizes={name: len(parts[name][0]) for name in SPLITS},
    )


def load_splits(
    cfg: Config, data_dir: str | Path | None = None, tfds_dir: str | Path | None = None
) -> Splits:
    """The image folder ``data_dir`` if given, otherwise tf_flowers (cached in ``tfds_dir``)."""
    if data_dir is not None:
        return load_folder_splits(data_dir, cfg)
    return load_tfds_splits(cfg, tfds_dir)


# --- Input pipeline ---------------------------------------------------------------------


def make_dataset(
    ds: tf.data.Dataset,
    num_classes: int,
    img_size: int,
    batch_size: int,
    training: bool = False,
    seed: int = 42,
    shuffle_buffer: int = 1000,
) -> tf.data.Dataset:
    """Resize, cast and one-hot encode, then batch. Only the training pipeline is shuffled.

    Pixels stay in the raw 0-255 range: ResNet-50's ``preprocess_input`` runs inside the
    model, after augmentation, so it is never applied twice.
    """

    def prepare(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.resize(image, [img_size, img_size])
        image.set_shape([img_size, img_size, 3])
        label = tf.one_hot(tf.cast(label, tf.int32), num_classes)  # one-hot for label smoothing
        return tf.cast(image, tf.float32), label

    ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
