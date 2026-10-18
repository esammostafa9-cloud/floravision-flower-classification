"""Load a trained model once and classify images from paths or raw bytes.

Shared by the CLI (``predict.py``) and the Gradio demo, so both preprocess images exactly like
training: decode to RGB, resize to the model's input size, raw 0-255 pixels. ResNet-50's
``preprocess_input`` is part of the saved model, so it is NOT applied here (applying it twice
was the bug in the first version of the app).

Uploaded images are untrusted, so they are checked before any pixels are decoded: file size,
image format, and pixel count (to stop decompression bombs). Only the parsers of the allowed
formats ever see the bytes.
"""

from __future__ import annotations

import hmac
import io
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from .download import MODEL_FILENAME, MODEL_SHA256, sha256_file
from .utils import load_json

MODEL_ENV = "FLORAVISION_MODEL_PATH"
DEFAULT_MODEL_PATH = Path("models") / MODEL_FILENAME

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_PIXELS = 25_000_000  # e.g. 6000 x 4000; about 75 MB once decoded to RGB
PARSERS = ("JPEG", "PNG", "WEBP", "BMP", "GIF")  # the only Pillow plugins that see the bytes
ALLOWED_FORMATS = (*PARSERS, "MPO")  # MPO: multi-picture JPEGs (phones), opened by JPEG


class InvalidImageError(ValueError):
    """The input is not an image we are willing to decode."""


def limit_pillow_decoding() -> None:
    """Make Pillow refuse images over ``MAX_IMAGE_PIXELS`` everywhere in this process.

    ``decode_image`` checks the size itself; this also covers code we don't control that
    decodes with Pillow, such as Gradio's own upload handling. By default Pillow only warns
    between 1x and 2x its limit, so the warning is turned into an error.
    """
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    warnings.simplefilter("error", Image.DecompressionBombWarning)


def decode_image(data: bytes) -> np.ndarray:
    """Validate and decode untrusted image bytes to a ``uint8`` RGB array ``[H, W, 3]``."""
    if not data:
        raise InvalidImageError("The file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImageError(f"The file is larger than {MAX_IMAGE_BYTES // 2**20} MB.")
    try:
        # formats= stops every other Pillow plugin from parsing the bytes at all.
        with Image.open(io.BytesIO(data), formats=PARSERS) as img:
            # Image.open only reads the header, so these checks run before any decoding.
            if img.format not in ALLOWED_FORMATS:
                raise InvalidImageError(f"Unsupported image format: {img.format}.")
            width, height = img.size
            if width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError(f"The image is too large ({width} x {height} pixels).")
            img = ImageOps.exif_transpose(img)  # respect camera orientation
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except InvalidImageError:
        raise
    except Exception as e:  # noqa: BLE001 - malformed files raise many types; all mean "reject"
        raise InvalidImageError("The file is not a readable image.") from e


def load_model_file(path: str | Path) -> keras.Model:
    """Load a ``.keras`` model for inference.

    ``safe_mode=True`` refuses models that would run arbitrary Python code (e.g. Lambda
    layers) on load. It is not a sandbox: only load model files you trust or have verified.
    """
    path = Path(path)
    if path.suffix != ".keras":
        raise ValueError(f"Expected a .keras model file, got {path.name}")
    return keras.models.load_model(path, compile=False, safe_mode=True)


def read_image_file(path: str | Path) -> bytes:
    """Read an image file, refusing anything over the size limit before loading it."""
    path = Path(path)
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise InvalidImageError(f"{path.name} is larger than {MAX_IMAGE_BYTES // 2**20} MB.")
    return path.read_bytes()


@dataclass
class Prediction:
    label: str
    confidence: float
    top_k: list[tuple[str, float]]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "top_k": [{"label": c, "probability": round(p, 4)} for c, p in self.top_k],
        }


class Classifier:
    """A trained model plus the class names its outputs refer to, in output order."""

    def __init__(self, model: keras.Model, class_names: list[str]):
        n_out = model.output_shape[-1]
        if n_out != len(class_names):
            raise ValueError(f"Model has {n_out} outputs but {len(class_names)} class names.")
        self.model = model
        self.class_names = list(class_names)
        self.img_size = int(model.input_shape[1])

    @classmethod
    def from_path(cls, model_path: str | Path, class_names_path: str | Path | None = None):
        """Load ``model.keras``; class names default to ``class_names.json`` beside it."""
        model_path = Path(model_path)
        names = load_json(class_names_path or model_path.parent / "class_names.json")
        return cls(load_model_file(model_path), names)

    @classmethod
    def from_env(cls):
        """Load the model named by ``FLORAVISION_MODEL_PATH``, or the downloaded model.

        The downloaded model's SHA-256 is checked again on every start, so a file swapped
        after download is refused.
        """
        path = Path(os.environ.get(MODEL_ENV) or DEFAULT_MODEL_PATH)
        if not path.is_file():
            raise RuntimeError(
                f"No model at {path}. Run `python -m floravision.download` first, or set "
                f"{MODEL_ENV} to a trained model, e.g. runs/<run>/model.keras"
            )
        if path.resolve() == DEFAULT_MODEL_PATH.resolve() and not hmac.compare_digest(
            sha256_file(path), MODEL_SHA256
        ):
            raise RuntimeError(
                f"{path} does not match the published model's SHA-256. Delete it and run "
                "`python -m floravision.download` again."
            )
        return cls.from_path(path)

    def _prepare(self, data: bytes) -> tf.Tensor:
        image = tf.image.resize(decode_image(data), [self.img_size, self.img_size])
        return tf.cast(image, tf.float32)

    def predict_proba(self, images: list[bytes]) -> np.ndarray:
        """Softmax probabilities, shape ``(len(images), num_classes)``."""
        batch = tf.stack([self._prepare(b) for b in images])
        logits = self.model(batch, training=False)
        return tf.nn.softmax(logits, axis=-1).numpy()

    def predict_bytes(self, data: bytes, top_k: int = 5) -> Prediction:
        """Classify one encoded image (JPEG / PNG / WebP / BMP / GIF bytes)."""
        return self.to_prediction(self.predict_proba([data])[0], top_k)

    def predict_path(self, path: str | Path, top_k: int = 5) -> Prediction:
        return self.predict_bytes(read_image_file(path), top_k)

    def to_prediction(self, prob: np.ndarray, top_k: int) -> Prediction:
        order = np.argsort(prob)[::-1][: max(1, min(top_k, len(prob)))]
        top = [(self.class_names[i], float(prob[i])) for i in order]
        return Prediction(label=top[0][0], confidence=top[0][1], top_k=top)
