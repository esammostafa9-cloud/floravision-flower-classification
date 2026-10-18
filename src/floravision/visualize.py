"""Plotting helpers. Every function writes a PNG and returns the figure.

Figures are built with Matplotlib's object-oriented API (no pyplot state), so they render
the same from the command line, in CI, or on a headless server.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import ConfusionMatrixDisplay


def _save(fig: Figure, path: str | Path) -> Figure:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_history(histories: dict[str, dict[str, list[float]]], path: str | Path) -> Figure:
    """Loss (with the gradient norm overlaid) and accuracy, all phases on one x-axis."""
    fig = Figure(figsize=(14, 5))
    ax_loss, ax_acc = fig.subplots(1, 2)

    series: dict[str, list[float]] = {
        k: [] for k in ("loss", "val_loss", "accuracy", "val_accuracy", "gradient_norm")
    }
    starts: list[tuple[int, str]] = []  # (first epoch of phase, phase name)
    for name, h in histories.items():
        starts.append((len(series["loss"]) + 1, name))
        for key, values in series.items():
            values.extend(h.get(key, [np.nan] * len(h["loss"])))
    epochs = np.arange(1, len(series["loss"]) + 1)

    ax_loss.plot(epochs, series["loss"], "-o", ms=3, label="train loss")
    ax_loss.plot(epochs, series["val_loss"], "-o", ms=3, label="val loss")
    ax_grad = ax_loss.twinx()
    ax_grad.plot(epochs, series["gradient_norm"], "--", color="crimson", alpha=0.6)
    ax_grad.set_ylabel("global gradient norm", color="crimson")
    ax_loss.plot([], [], "--", color="crimson", alpha=0.6, label="gradient norm")

    ax_acc.plot(epochs, series["accuracy"], "-o", ms=3, label="train accuracy")
    ax_acc.plot(epochs, series["val_accuracy"], "-o", ms=3, label="val accuracy")

    for ax, title in ((ax_loss, "Loss"), (ax_acc, "Accuracy")):
        for start, name in starts[1:]:
            ax.axvline(start - 0.5, color="grey", ls="--", lw=1)
            ax.text(start - 0.3, 0.02, name, transform=ax.get_xaxis_transform(), color="grey")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    return _save(fig, path)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], path: str | Path, title: str = "Confusion matrix"
) -> Figure:
    """Row-normalised confusion matrix: each row shows where one true class's images went."""
    fig = Figure(figsize=(7, 6))
    ax = fig.add_subplot()
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, values_format=".2f", colorbar=True
    )
    ax.set_title(f"{title} (normalised by true class)", fontsize=12, pad=12)
    return _save(fig, path)


def plot_confident_mistakes(
    images: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    path: str | Path,
    k: int = 3,
) -> Figure:
    """The ``k`` misclassified images the model was most confident about."""
    y_pred = probs.argmax(axis=1)
    wrong = np.flatnonzero(y_pred != y_true)
    chosen = wrong[np.argsort(probs[wrong].max(axis=1))[::-1][:k]]

    fig = Figure(figsize=(4 * max(1, len(chosen)), 4))
    if len(chosen) == 0:
        ax = fig.add_subplot()
        ax.text(0.5, 0.5, "No misclassified images", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    for i, idx in enumerate(chosen):
        ax = fig.add_subplot(1, len(chosen), i + 1)
        ax.imshow(images[idx])
        ax.axis("off")
        ax.set_title(
            f"True: {class_names[y_true[idx]]}\n"
            f"Pred: {class_names[y_pred[idx]]} ({probs[idx].max():.0%})",
            fontsize=10,
        )
    fig.suptitle("Most confident mistakes")
    return _save(fig, path)


def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    path: str | Path,
) -> Figure:
    """One row per image: the image, then a bar chart of its softmax probabilities."""
    n = len(images)
    fig = Figure(figsize=(9, 2.6 * n))
    axes = np.atleast_2d(fig.subplots(n, 2, gridspec_kw={"width_ratios": [1, 1.6]}))

    for row, (image, true, prob) in enumerate(zip(images, y_true, probs, strict=True)):
        pred = int(prob.argmax())
        colour = "green" if pred == int(true) else "red"

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"True: {class_names[int(true)]}", fontsize=10)
        axes[row, 0].axis("off")

        ypos = np.arange(len(class_names))[::-1]
        colours = ["#4C72B0"] * len(class_names)
        colours[pred] = colour
        axes[row, 1].barh(ypos, prob, color=colours)
        axes[row, 1].set_yticks(ypos, class_names, fontsize=9)
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].set_xlabel("Softmax probability")
        axes[row, 1].set_title(
            f"Pred: {class_names[pred]} ({prob[pred]:.2%})", fontsize=10, color=colour
        )
    return _save(fig, path)
