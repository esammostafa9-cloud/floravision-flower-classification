"""Inference, input validation and the Gradio demo, using a small untrained model."""

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from floravision import inference
from floravision.config import Config
from floravision.inference import Classifier, InvalidImageError, decode_image
from floravision.model import build_model

CLASSES = ["dandelion", "daisy", "tulips"]
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    cfg = Config(img_size=32, pretrained_weights=None, dense_units=8)
    model, _ = build_model(len(CLASSES), cfg)
    out = tmp_path_factory.mktemp("model")
    model.save(out / "model.keras")
    (out / "class_names.json").write_text(json.dumps(CLASSES))
    return out


@pytest.fixture(scope="module")
def classifier(model_dir):
    return Classifier.from_path(model_dir / "model.keras")


def image_bytes(fmt="JPEG", size=(48, 40), mode="RGB") -> bytes:
    buf = io.BytesIO()
    pixels = np.random.default_rng(0).integers(0, 255, (*size[::-1], 3), dtype=np.uint8)
    Image.fromarray(pixels).convert(mode).save(buf, format=fmt)
    return buf.getvalue()


# --- inference -------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "BMP"])
def test_predict_bytes_returns_sorted_top_k(classifier, fmt):
    pred = classifier.predict_bytes(image_bytes(fmt), top_k=2)
    assert pred.label in CLASSES
    assert len(pred.top_k) == 2
    probs = [p for _, p in pred.top_k]
    assert probs == sorted(probs, reverse=True)
    assert pred.confidence == probs[0]


def test_probabilities_are_softmax_of_logits(classifier):
    pred = classifier.predict_bytes(image_bytes(), top_k=99)
    assert len(pred.top_k) == len(CLASSES)  # top_k is clamped
    assert sum(p for _, p in pred.top_k) == pytest.approx(1.0, abs=1e-5)


def test_inference_matches_training_preprocessing(classifier):
    """Regression test for the original app, which applied preprocess_input a second time.

    The classifier must feed the model raw, resized 0-255 pixels: the same tensor the
    training pipeline produces.
    """
    data = image_bytes("PNG", size=(64, 50))  # lossless, so both decoders agree exactly
    raw = tf.io.decode_image(data, channels=3, expand_animations=False)
    expected = tf.nn.softmax(classifier.model(tf.image.resize(raw, [32, 32])[None], training=False))
    np.testing.assert_allclose(classifier.predict_proba([data])[0], expected.numpy()[0], atol=1e-3)


def test_class_names_follow_the_saved_order(classifier):
    assert classifier.class_names == CLASSES  # not re-sorted


def test_multi_picture_jpeg_from_phones_is_accepted(classifier):
    first = Image.open(io.BytesIO(image_bytes()))
    buf = io.BytesIO()
    first.save(buf, format="MPO", save_all=True, append_images=[first.copy()])
    assert Image.open(io.BytesIO(buf.getvalue())).format == "MPO"
    assert classifier.predict_bytes(buf.getvalue()).label in CLASSES


def test_grayscale_and_rgba_images_are_converted(classifier):
    for mode, fmt in (("L", "PNG"), ("RGBA", "PNG"), ("P", "GIF")):
        assert classifier.predict_bytes(image_bytes(fmt, mode=mode)).label in CLASSES


def test_class_name_mismatch_is_rejected(classifier):
    with pytest.raises(ValueError, match="outputs"):
        Classifier(classifier.model, [*CLASSES, "extra"])


def test_from_env_loads_model_path(model_dir, monkeypatch):
    monkeypatch.setenv(inference.MODEL_ENV, str(model_dir / "model.keras"))
    assert Classifier.from_env().class_names == CLASSES


def test_from_env_explains_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv(inference.MODEL_ENV, str(tmp_path / "missing.keras"))
    with pytest.raises(RuntimeError, match="floravision.train"):
        Classifier.from_env()


# --- untrusted input --------------------------------------------------------------------


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50])
def test_invalid_bytes_are_rejected(data):
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_oversized_file_is_rejected_before_decoding(monkeypatch):
    monkeypatch.setattr(inference, "MAX_IMAGE_BYTES", 100)
    with pytest.raises(InvalidImageError, match="larger than"):
        decode_image(image_bytes())


def test_too_many_pixels_is_rejected(monkeypatch):
    """Decompression-bomb guard: the pixel count is checked from the header alone."""
    monkeypatch.setattr(inference, "MAX_IMAGE_PIXELS", 40 * 40)
    with pytest.raises(InvalidImageError, match="too large"):
        decode_image(image_bytes(size=(48, 40)))


@pytest.mark.parametrize("fmt", ["TIFF", "PPM", "ICO", "TGA"])
def test_disallowed_formats_never_reach_a_parser(fmt):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format=fmt)
    with pytest.raises(InvalidImageError):
        decode_image(buf.getvalue())


def test_any_decoder_failure_becomes_invalid_image(monkeypatch):
    """Malformed files make Pillow raise struct.error, ValueError, etc.; all are rejections."""
    import struct

    def broken(img):
        raise struct.error("bad EXIF")

    monkeypatch.setattr(inference.ImageOps, "exif_transpose", broken)
    with pytest.raises(InvalidImageError, match="not a readable image"):
        decode_image(image_bytes("WEBP"))


def test_truncated_image_is_rejected():
    with pytest.raises(InvalidImageError):
        decode_image(image_bytes("PNG", size=(64, 64))[:200])


def test_limit_pillow_decoding_also_guards_other_pillow_users(monkeypatch):
    """Code we don't control (e.g. Gradio) decodes with Pillow too; it must hit the limit."""
    import warnings

    monkeypatch.setattr(inference, "MAX_IMAGE_PIXELS", 40 * 40)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    with warnings.catch_warnings():
        inference.limit_pillow_decoding()
        with pytest.raises((Image.DecompressionBombError, Image.DecompressionBombWarning)):
            Image.open(io.BytesIO(image_bytes("PNG", size=(48, 40)))).load()


def test_read_image_file_checks_size_before_reading(tmp_path, monkeypatch):
    big = tmp_path / "big.jpg"
    big.write_bytes(image_bytes())
    monkeypatch.setattr(inference, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(InvalidImageError, match="larger than"):
        inference.read_image_file(big)


def test_model_with_lambda_layer_is_refused(tmp_path):
    """safe_mode blocks models that would execute arbitrary Python when loaded."""
    import keras

    inputs = keras.Input((4,))
    outputs = keras.layers.Lambda(lambda x: x * 2)(inputs)
    keras.Model(inputs, outputs).save(tmp_path / "lambda.keras")
    (tmp_path / "class_names.json").write_text(json.dumps(["a", "b", "c", "d"]))
    with pytest.raises(ValueError, match="(?i)lambda|safe_mode"):
        Classifier.from_path(tmp_path / "lambda.keras")


def test_only_keras_files_are_loaded(tmp_path):
    (tmp_path / "model.pkl").write_bytes(b"not a model")
    with pytest.raises(ValueError, match=r"\.keras"):
        inference.load_model_file(tmp_path / "model.pkl")


def test_from_env_falls_back_to_the_default_model_path(model_dir, tmp_path, monkeypatch):
    default = tmp_path / "floravision_resnet50.keras"
    default.write_bytes((model_dir / "model.keras").read_bytes())
    (tmp_path / "class_names.json").write_text(json.dumps(CLASSES))
    monkeypatch.delenv(inference.MODEL_ENV, raising=False)
    monkeypatch.setattr(inference, "DEFAULT_MODEL_PATH", default)
    assert Classifier.from_env().class_names == CLASSES


# --- Gradio demo -----------------------------------------------------------------------


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("demo_app", REPO_ROOT / "demo" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_builds_and_has_one_example_per_class(classifier):
    import gradio as gr

    demo_app = _load_demo_module()
    assert isinstance(demo_app.build_demo(classifier), gr.Blocks)
    examples = {p.stem for p in (REPO_ROOT / "demo" / "examples").glob("*.jpg")}
    assert examples == {"daisy", "dandelion", "roses", "sunflowers", "tulips"}


def test_result_card_flags_low_confidence():
    demo_app = _load_demo_module()
    assert "unsure" not in demo_app.result_card("sunflowers", 0.97, "dandelion")
    unsure = demo_app.result_card("roses", 0.41, "tulips")
    assert "unsure" in unsure
    assert "Tulips" in unsure


def test_cards_escape_html():
    demo_app = _load_demo_module()
    payload = "<script>alert(1)</script>"
    assert "<script>" not in demo_app.result_card(payload, 0.9, None)
    assert "<script>" not in demo_app.error_card(payload)


def test_upload_component_does_not_decode_images(classifier):
    """image_mode=None makes Gradio pass the file through untouched (no decode/re-encode),
    so the size and pixel checks in floravision.inference run first."""
    import gradio as gr

    demo_app = _load_demo_module()
    images = [b for b in demo_app.build_demo(classifier).blocks.values() if isinstance(b, gr.Image)]
    assert images and all(b.image_mode is None and b.type == "filepath" for b in images)
    assert demo_app.MAX_REQUEST_BYTES >= inference.MAX_IMAGE_BYTES * 4 // 3  # base64 fits


@pytest.fixture(scope="module")
def client(classifier):
    from fastapi.testclient import TestClient

    with TestClient(_load_demo_module().create_app(classifier)) as c:
        yield c


def test_app_serves_the_demo_without_api_docs(client):
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 404


def test_oversized_request_bodies_are_rejected(client):
    """Covers base64 data: URLs in JSON, which Gradio's max_file_size does not limit."""
    demo_app = _load_demo_module()
    big = b"{" + b" " * (demo_app.MAX_REQUEST_BYTES + 1) + b"}"
    r = client.post(
        "/gradio_api/queue/join", content=big, headers={"content-type": "application/json"}
    )
    assert r.status_code == 413


def test_oversized_upload_is_rejected(client):
    r = client.post(
        "/gradio_api/upload",
        files={"files": ("big.jpg", b"0" * (inference.MAX_IMAGE_BYTES + 1), "image/jpeg")},
    )
    assert r.status_code == 413


def test_chunked_bodies_without_length_are_capped():
    import asyncio

    demo_app = _load_demo_module()
    seen = []

    async def app(scope, receive, send):
        while (await receive()).get("more_body"):
            pass
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def run():
        chunks = [{"type": "http.request", "body": b"x" * 600, "more_body": True}] * 3

        async def receive():
            return chunks.pop(0)

        async def send(message):
            seen.append(message)

        middleware = demo_app.BodySizeLimit(app, max_bytes=1000)
        await middleware({"type": "http", "headers": []}, receive, send)

    asyncio.run(run())
    assert seen[0]["status"] == 413
