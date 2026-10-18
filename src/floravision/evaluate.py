"""Evaluate a trained model on the held-out test split.

Usage:
    python -m floravision.evaluate --model models/floravision_resnet50.keras
    python -m floravision.evaluate --model runs/<run>/model.keras --data-dir my_flowers/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import Config
from .data import align_labels, load_splits, make_dataset
from .inference import load_model_file
from .model import make_loss
from .utils import get_logger, load_json, save_json
from .visualize import plot_confident_mistakes, plot_confusion_matrix, plot_sample_predictions

log = get_logger()


def collect_predictions(
    model: keras.Model, ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a batched dataset; return uint8 images, one-hot labels, logits."""
    images, labels, logits = [], [], []
    for x, y in ds:
        logits.append(model(x, training=False).numpy())
        images.append(np.clip(x.numpy(), 0, 255).astype(np.uint8))
        labels.append(y.numpy())
    return np.concatenate(images), np.concatenate(labels), np.concatenate(logits)


def evaluate_model(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    cfg: Config,
    out_dir: str | Path,
    n_samples: int = 5,
) -> dict[str, float]:
    """Score ``model`` once on the (unbatched) test split and write metrics and figures.

    ``test_ds`` labels must already be in the model's class order (see ``align_labels``).
    """
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    batched = make_dataset(test_ds, len(class_names), cfg.img_size, cfg.batch_size)
    images, onehot, logits = collect_predictions(model, batched)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    y_true, y_pred = onehot.argmax(axis=1), probs.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        # Same loss as training (label smoothing 0.1), so it has a floor of about 0.4.
        "test_loss": float(make_loss(cfg)(onehot, logits)),
        "test_images": len(y_true),
    }
    metrics = {k: round(float(v), 4) if k != "test_images" else v for k, v in metrics.items()}
    save_json(metrics, out_dir / "test_metrics.json")

    all_labels = list(range(len(class_names)))
    report_kwargs = dict(labels=all_labels, target_names=class_names, zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, **report_kwargs)
    pd.DataFrame(report).transpose().round(4).to_csv(out_dir / "classification_report.csv")
    log.info("\n%s", classification_report(y_true, y_pred, digits=3, **report_kwargs))

    pd.DataFrame(
        {
            "index": np.arange(len(y_true)),
            "true": [class_names[i] for i in y_true],
            "pred": [class_names[i] for i in y_pred],
            "confidence": probs.max(axis=1).round(4),
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize="true")
    plot_confusion_matrix(cm, class_names, fig_dir / "confusion_matrix.png")
    plot_confident_mistakes(images, y_true, probs, class_names, fig_dir / "confident_mistakes.png")
    rng = np.random.default_rng(cfg.seed)
    sample = rng.choice(len(y_true), size=min(n_samples, len(y_true)), replace=False)
    plot_sample_predictions(
        images[sample],
        y_true[sample],
        probs[sample],
        class_names,
        fig_dir / "sample_predictions.png",
    )

    log.info("Test metrics: %s", metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, type=Path, help="Trained .keras model")
    parser.add_argument(
        "--class-names", type=Path, help="class_names.json (default: next to the model)"
    )
    parser.add_argument(
        "--config", type=Path, help="YAML config (default: config.yaml next to the model)"
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Folder of <class>/*.jpg images (default: tf_flowers)"
    )
    parser.add_argument("--tfds-dir", type=Path, help="TensorFlow Datasets cache folder")
    parser.add_argument(
        "--output-dir", type=Path, help="Where to write results (default: the model's folder)"
    )
    args = parser.parse_args(argv)

    config_path = args.config or args.model.parent / "config.yaml"
    cfg = Config.from_yaml(config_path if config_path.is_file() else None)
    class_names = load_json(args.class_names or args.model.parent / "class_names.json")

    splits = load_splits(cfg, args.data_dir, args.tfds_dir)
    test_ds = align_labels(splits.test, splits.class_names, class_names)
    model = load_model_file(args.model)
    evaluate_model(model, test_ds, class_names, cfg, args.output_dir or args.model.parent)


if __name__ == "__main__":
    main()
