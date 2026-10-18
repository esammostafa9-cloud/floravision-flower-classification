"""Classify one or more flower images with a trained model.

Usage:
    python -m floravision.predict image.jpg [more images or folders]
    python -m floravision.predict --model runs/<run>/model.keras my_folder/ --output preds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import IMAGE_EXTS
from .inference import DEFAULT_MODEL_PATH, Classifier, InvalidImageError, read_image_file
from .utils import get_logger

log = get_logger()


def collect_images(inputs: list[Path]) -> list[Path]:
    """Expand folders into the image files they contain (non-recursive), keep files as-is."""
    images = []
    for p in inputs:
        if p.is_dir():
            images.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            images.append(p)
        else:
            raise FileNotFoundError(p)
    return images


def predict(classifier: Classifier, images: list[Path], top_k: int = 3) -> pd.DataFrame:
    rows = []
    for path in images:
        try:
            pred = classifier.predict_bytes(read_image_file(path), top_k)
        except InvalidImageError as e:
            log.warning("Skipping %s: %s", path, e)
            continue
        row: dict[str, str | float] = {"image": str(path)}
        for rank, (label, p) in enumerate(pred.top_k, start=1):
            row[f"top{rank}"] = label
            row[f"top{rank}_prob"] = round(p, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files and/or folders")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Trained .keras model (default: {DEFAULT_MODEL_PATH.as_posix()})",
    )
    parser.add_argument(
        "--class-names", type=Path, help="class_names.json (default: next to the model)"
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Optional CSV file for the predictions")
    args = parser.parse_args(argv)

    images = collect_images(args.inputs)
    if not images:
        parser.error("no images found")
    classifier = Classifier.from_path(args.model, args.class_names)

    results = predict(classifier, images, args.top_k)
    print(results.to_string(index=False))
    if args.output:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
