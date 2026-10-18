"""Two-phase transfer-learning training on tf_flowers.

Phase 1 trains only the classification head on a frozen ResNet-50. Phase 2 unfreezes the last
layers of the backbone and fine-tunes them at a much lower learning rate. The final model is
the phase checkpoint with the lowest *validation* loss; the test split is used only once, at
the end.

Usage:
    python -m floravision.train                         # tf_flowers, downloaded on first run
    python -m floravision.train --data-dir my_flowers/  # your own <class>/*.jpg folders
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import keras
import tensorflow as tf

from .config import Config
from .data import Splits, load_splits, make_dataset
from .evaluate import evaluate_model
from .inference import load_model_file
from .model import build_model, compile_model, make_loss, unfreeze_last_layers
from .utils import get_logger, save_json, set_seed
from .visualize import plot_history

log = get_logger()


class GradientNormLogger(keras.callbacks.Callback):
    """Record the global gradient norm once per epoch on a fixed training batch.

    A steady, low norm means optimisation is well behaved; spikes point to instability.
    """

    def __init__(self, sample_batch: tuple[tf.Tensor, tf.Tensor], loss: keras.losses.Loss):
        super().__init__()
        self.images, self.labels = sample_batch
        self.loss = loss
        self.norms: list[float] = []

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        with tf.GradientTape() as tape:
            logits = self.model(self.images, training=True)
            loss = self.loss(self.labels, logits)
            if self.model.losses:
                loss = loss + tf.add_n(self.model.losses)
        grads = [g for g in tape.gradient(loss, self.model.trainable_weights) if g is not None]
        self.norms.append(float(tf.linalg.global_norm(grads)))
        if logs is not None:
            logs["gradient_norm"] = self.norms[-1]


def _callbacks(name: str, out_dir: Path, cfg: Config, grad_logger: GradientNormLogger) -> list:
    patience = cfg.head_patience if name == "phase1_head" else cfg.fine_tune_patience
    callbacks = [
        grad_logger,  # first, so gradient_norm reaches the CSV log
        keras.callbacks.ModelCheckpoint(
            out_dir / f"{name}_best.keras", monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.CSVLogger(out_dir / f"history_{name}.csv"),
    ]
    if name == "phase1_head":
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.head_lr_factor,
                patience=cfg.lr_patience,
                min_lr=cfg.head_min_lr,
                verbose=1,
            )
        )
    return callbacks


def train(splits: Splits, cfg: Config, out_dir: Path, run_eval: bool = True) -> Path:
    """Run both phases on ``splits`` and return the path of the selected model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    cfg.to_yaml(out_dir / "config.yaml")
    save_json(splits.class_names, out_dir / "class_names.json")
    log.info("Classes (model output order): %s", splits.class_names)
    log.info("Split sizes: %s", splits.sizes)

    pipelines = {
        split: make_dataset(
            splits[split],
            splits.num_classes,
            cfg.img_size,
            cfg.batch_size,
            training=split == "train",
            seed=cfg.seed,
            shuffle_buffer=cfg.shuffle_buffer,
        )
        for split in ("train", "val")
    }
    sample_batch = next(iter(pipelines["train"]))

    model, base_model = build_model(splits.num_classes, cfg)
    phases = [
        ("phase1_head", cfg.head_epochs, cfg.head_lr),
        ("phase2_finetune", cfg.fine_tune_epochs, cfg.fine_tune_lr),
    ]

    histories, candidates = {}, []
    for name, epochs, lr in phases:
        if epochs <= 0:
            log.info("Skipping %s (0 epochs)", name)
            continue
        if name == "phase2_finetune":
            unfreeze_last_layers(base_model, cfg.fine_tune_layers)
        compile_model(model, lr, cfg)
        log.info("%s: %d trainable weight tensors, lr=%g", name, len(model.trainable_weights), lr)

        grad_logger = GradientNormLogger(sample_batch, make_loss(cfg))
        history = model.fit(
            pipelines["train"],
            validation_data=pipelines["val"],
            epochs=epochs,
            callbacks=_callbacks(name, out_dir, cfg, grad_logger),
        )
        h = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        h["gradient_norm"] = grad_logger.norms
        histories[name] = h
        best_val_loss = min(h["val_loss"])
        candidates.append((best_val_loss, name, out_dir / f"{name}_best.keras"))
        log.info("%s best val loss: %.4f", name, best_val_loss)

    if not candidates:
        raise ValueError("Both phases have 0 epochs; nothing to train.")

    save_json(histories, out_dir / "history.json")
    plot_history(histories, out_dir / "figures" / "training_curves.png")

    # Model selection on validation loss only, as in the notebook's checkpoints.
    best_val_loss, best_name, best_ckpt = min(candidates)
    final_path = out_dir / "model.keras"
    shutil.copyfile(best_ckpt, final_path)
    save_json(
        {"selected_phase": best_name, "val_loss": round(best_val_loss, 4)},
        out_dir / "selection.json",
    )
    log.info("Selected %s (val loss %.4f) -> %s", best_name, best_val_loss, final_path)

    if run_eval:
        final_model = load_model_file(final_path)
        evaluate_model(final_model, splits.test, splits.class_names, cfg, out_dir)

    return final_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Folder of <class>/*.jpg images (default: tf_flowers)"
    )
    parser.add_argument("--tfds-dir", type=Path, help="TensorFlow Datasets cache folder")
    parser.add_argument("--config", type=Path, help="YAML config (default: built-in defaults)")
    parser.add_argument("--output-dir", type=Path, help="Run folder (default: runs/<timestamp>)")
    parser.add_argument("--head-epochs", type=int, help="Override phase 1 epoch budget")
    parser.add_argument("--fine-tune-epochs", type=int, help="Override phase 2 epoch budget")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument(
        "--skip-eval", action="store_true", help="Do not evaluate on the test split"
    )
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(
        args.config,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
    )
    splits = load_splits(cfg, args.data_dir, args.tfds_dir)
    out_dir = args.output_dir or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    train(splits, cfg, out_dir, run_eval=not args.skip_eval)


if __name__ == "__main__":
    main()
