"""Gradio demo: classify a flower photo as daisy, dandelion, rose, sunflower or tulip.

Usage:
    FLORAVISION_MODEL_PATH=runs/<run>/model.keras python demo/app.py
    # then open http://127.0.0.1:7860

Without FLORAVISION_MODEL_PATH, the model is loaded from models/floravision_resnet50.keras.
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI

from floravision.inference import (
    MAX_IMAGE_BYTES,
    Classifier,
    InvalidImageError,
    limit_pillow_decoding,
)

HERE = Path(__file__).parent
GITHUB_URL = "https://github.com/esammostafa9-cloud/floravision-flower-classification"
UNSURE_THRESHOLD = 0.5
# A 10 MB image is about 13.4 MB once base64-encoded into a JSON request.
MAX_REQUEST_BYTES = 15 * 1024 * 1024
# Gradio follows the visitor's system theme; the demo is designed for dark mode only.
FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"

METRICS = {
    "Test accuracy": "92.5%",
    "Macro F1": "0.926",
    "Classes": "5",
    "Test images": "550",
}

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
#header { text-align: center; }
#header h1 { margin-bottom: 0.2rem; }
#header p { color: var(--body-text-color-subdued); margin-top: 0; }
.result-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    padding: 20px 24px;
    background: var(--block-background-fill);
}
.result-card .label { font-size: 1.9rem; font-weight: 700; line-height: 1.2; }
.result-card .caption { color: var(--body-text-color-subdued); font-size: 0.9rem; }
.badge {
    display: inline-block; margin-top: 10px; padding: 4px 12px; border-radius: 999px;
    font-weight: 600; font-size: 0.9rem;
}
.badge.sure { background: rgba(34, 197, 94, 0.15); color: rgb(22, 163, 74); }
.badge.unsure { background: rgba(245, 158, 11, 0.18); color: rgb(217, 119, 6); }
.badge.error { background: rgba(239, 68, 68, 0.15); color: rgb(220, 38, 38); }
.metrics { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin: 8px 0 4px; }
.metric {
    border: 1px solid var(--border-color-primary); border-radius: 12px;
    padding: 8px 16px; text-align: center; min-width: 110px;
}
.metric b { display: block; font-size: 1.25rem; }
.metric span { color: var(--body-text-color-subdued); font-size: 0.8rem; }
"""


def pretty(label: str) -> str:
    return html.escape(label.replace("_", " ").capitalize())


def result_card(label: str, confidence: float, runner_up: str | None) -> str:
    sure = confidence >= UNSURE_THRESHOLD
    badge = (
        f'<span class="badge sure">{confidence:.1%} confident</span>'
        if sure
        else f'<span class="badge unsure">{confidence:.1%} — the model is unsure</span>'
    )
    hint = (
        f'<div class="caption" style="margin-top:8px">Could also be: {pretty(runner_up)}</div>'
        if runner_up and not sure
        else ""
    )
    return (
        '<div class="result-card">'
        '<div class="caption">Predicted flower</div>'
        f'<div class="label">{pretty(label)}</div>{badge}{hint}</div>'
    )


def error_card(message: str) -> str:
    return (
        '<div class="result-card"><div class="caption">Predicted flower</div>'
        '<div class="label">—</div>'
        f'<span class="badge error">{html.escape(message)}</span></div>'
    )


EMPTY_CARD = (
    '<div class="result-card"><div class="caption">Predicted flower</div>'
    '<div class="label">—</div>'
    '<div class="caption" style="margin-top:8px">Upload a photo or pick an example.</div></div>'
)


def build_demo(classifier: Classifier) -> gr.Blocks:
    def classify(image_path: str | None):
        if not image_path:
            return EMPTY_CARD, None
        try:
            pred = classifier.predict_path(image_path, top_k=5)
        except InvalidImageError as e:
            return error_card(str(e)), None
        runner_up = pred.top_k[1][0] if len(pred.top_k) > 1 else None
        scores = {pretty(c): p for c, p in pred.top_k}
        return result_card(pred.label, pred.confidence, runner_up), scores

    examples = sorted(str(p) for p in (HERE / "examples").glob("*.jpg"))
    metrics_html = (
        '<div class="metrics">'
        + "".join(
            f'<div class="metric"><b>{v}</b><span>{k}</span></div>' for k, v in METRICS.items()
        )
        + "</div>"
    )
    classes = ", ".join(pretty(c) for c in sorted(classifier.class_names))

    # Analytics off; uploaded files are deleted from the cache after an hour.
    with gr.Blocks(title="FloraVision", analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
        gr.HTML(
            '<div id="header"><h1>🌸 FloraVision</h1>'
            "<p>Recognise five common flowers from a single photo — "
            "ResNet-50 fine-tuned on tf_flowers</p></div>",
            padding=False,
        )
        gr.HTML(metrics_html, padding=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                image = gr.Image(
                    type="filepath",
                    # None: Gradio passes the file through without decoding or re-encoding
                    # it, so floravision.inference validates the bytes before anything else.
                    image_mode=None,
                    label="Flower photo",
                    sources=["upload", "clipboard"],
                    height=360,
                )
                with gr.Row():
                    clear = gr.Button("Clear", variant="secondary")
                    submit = gr.Button("Classify", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=image, label="Try an example")
            with gr.Column(scale=4):
                card = gr.HTML(EMPTY_CARD, padding=False)
                scores = gr.Label(
                    num_top_classes=5, label="Class probabilities", show_heading=False
                )

        with gr.Accordion("About this model", open=False):
            gr.Markdown(
                f"""
**Model.** ResNet-50 pretrained on ImageNet, trained in two phases: first a small
classification head on the frozen backbone, then fine-tuning of the backbone's last 15
layers at a very low learning rate (BatchNorm frozen).

**Data.** [tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) — 3,670
photos, split 70 / 15 / 15. Metrics above are on the 550 held-out test images.

**Known weak spots.** Roses and tulips are confused most often: both have layered,
colourful petals.

**Tip.** The model only knows these 5 flowers. Any other flower (or a non-flower photo)
will still be assigned to one of them, sometimes with high confidence. It is a demo: never
use it to decide whether a plant is safe to eat or touch.

**Classes:** {classes}

**Example photos** (Flickr, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)):
daisy by [Andrés Moreira](https://www.flickr.com/photos/andrix/8708143485/),
dandelion by [Andrey](https://www.flickr.com/photos/akras/2693136371/),
roses by [Olga Khomitsevich](https://www.flickr.com/photos/zemzina/488849503/),
sunflowers by [Stephane Mignon](https://www.flickr.com/photos/topsteph53/8174935717/),
tulips by [jsogo](https://www.flickr.com/photos/jsogo/8904780994/).

Code: [GitHub]({GITHUB_URL})
"""
            )

        # Kept off the API docs page. This is not access control: the browser UI calls the
        # same HTTP endpoint, so input validation in floravision.inference is what protects it.
        submit.click(classify, inputs=image, outputs=[card, scores], api_visibility="private")
        image.change(classify, inputs=image, outputs=[card, scores], api_visibility="private")
        clear.click(
            lambda: (None, EMPTY_CARD, None),
            outputs=[image, card, scores],
            api_visibility="private",
        )

    return demo


class BodySizeLimit:
    """ASGI middleware that rejects request bodies over ``max_bytes`` with 413.

    Gradio's ``max_file_size`` only covers its upload route. Images can also arrive as base64
    ``data:`` URLs inside JSON, so every request body is capped here.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > self.max_bytes):
            return await self._reject(send)

        received, started = 0, False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:  # chunked bodies have no content-length
                    raise _BodyTooLarge
            return message

        async def tracking_send(message):
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(send)

    @staticmethod
    async def _reject(send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"Request body too large"})


class _BodyTooLarge(Exception):
    pass


def create_app(classifier: Classifier) -> FastAPI:
    """The Gradio demo mounted on a FastAPI app that caps request sizes."""
    demo = build_demo(classifier)
    demo.queue(default_concurrency_limit=2, max_size=20)  # bound CPU and memory use

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodySizeLimit, max_bytes=MAX_REQUEST_BYTES)
    return gr.mount_gradio_app(
        app,
        demo,
        path="/",
        show_error=False,  # never send exception details to the browser
        max_file_size=MAX_IMAGE_BYTES,
        footer_links=["gradio"],
        mcp_server=False,
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="rose"),
        css=CSS,
        js=FORCE_DARK_JS,
    )


def main() -> None:
    limit_pillow_decoding()  # also applies to Gradio's own image handling
    uvicorn.run(
        create_app(Classifier.from_env()),
        # Local only by default. Set GRADIO_SERVER_NAME=0.0.0.0 to serve on your network.
        host=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        server_header=False,
    )


if __name__ == "__main__":
    main()
