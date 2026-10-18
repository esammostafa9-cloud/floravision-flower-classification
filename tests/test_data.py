import numpy as np
import pytest
import tensorflow as tf
from conftest import CLASSES, PER_CLASS

from floravision.data import (
    align_labels,
    list_image_folder,
    load_folder_splits,
    make_dataset,
    tfds_split_specs,
)


def test_tfds_split_specs_match_the_notebook():
    assert tfds_split_specs(0.15, 0.15) == ["train[:70%]", "train[70%:85%]", "train[85%:]"]
    with pytest.raises(ValueError):
        tfds_split_specs(0.0, 0.0)


def test_list_image_folder_collects_images_only(tiny_dataset):
    paths, labels, class_names = list_image_folder(tiny_dataset)
    assert class_names == CLASSES
    assert len(paths) == len(CLASSES) * PER_CLASS
    assert np.bincount(labels).tolist() == [PER_CLASS] * len(CLASSES)
    assert not any(p.endswith(".txt") for p in paths)


def test_list_image_folder_requires_class_folders(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_image_folder(tmp_path / "missing")
    (tmp_path / "only_one").mkdir()
    with pytest.raises(ValueError, match="at least 2"):
        list_image_folder(tmp_path)


def _labels(ds: tf.data.Dataset) -> list[int]:
    return [int(y) for _, y in ds]


def test_folder_split_is_stratified_disjoint_and_deterministic(tiny_dataset, tiny_config):
    splits = load_folder_splits(tiny_dataset, tiny_config)
    assert splits.sizes == {"train": 18, "val": 6, "test": 6}
    for name in ("train", "val", "test"):
        counts = np.bincount(_labels(splits[name]), minlength=len(CLASSES))
        assert len(set(counts.tolist())) == 1, f"{name} split is not stratified: {counts}"

    again = load_folder_splits(tiny_dataset, tiny_config)
    assert _labels(splits.test) == _labels(again.test)


def test_pipeline_yields_raw_pixels_and_one_hot_labels(tiny_dataset, tiny_config):
    splits = load_folder_splits(tiny_dataset, tiny_config)
    batch = make_dataset(splits.train, splits.num_classes, img_size=32, batch_size=4, training=True)
    images, labels = next(iter(batch))
    assert tuple(images.shape) == (4, 32, 32, 3)
    assert images.dtype == tf.float32
    assert float(tf.reduce_max(images)) > 1.0  # still 0-255: preprocess_input is in the model
    assert tuple(labels.shape) == (4, len(CLASSES))
    np.testing.assert_allclose(labels.numpy().sum(axis=1), 1.0)


def test_align_labels_maps_into_model_order():
    data_names = ["daisy", "dandelion", "roses"]  # e.g. sorted folder names
    model_names = ["dandelion", "daisy", "roses"]  # e.g. tf_flowers order
    ds = tf.data.Dataset.from_tensor_slices((tf.zeros((3, 2, 2, 3)), [0, 1, 2]))
    aligned = align_labels(ds, data_names, model_names)
    assert [model_names[y] for y in _labels(aligned)] == data_names


def test_align_labels_rejects_different_classes():
    ds = tf.data.Dataset.from_tensor_slices((tf.zeros((1, 2, 2, 3)), [0]))
    with pytest.raises(ValueError, match="do not match"):
        align_labels(ds, ["daisy", "roses"], ["daisy", "tulips"])
