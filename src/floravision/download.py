"""Download the published model and check its SHA-256 before it is used.

A ``.keras`` file is code-adjacent: loading a tampered one is a security risk. The download is
only kept if its hash matches the one pinned below, so a corrupted or swapped file is never
written to the model path.

Usage:
    python -m floravision.download                 # -> models/floravision_resnet50.keras
    python -m floravision.download --output-dir my_models
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .utils import get_logger, save_json

log = get_logger()

GITHUB_REPO = "esammostafa9-cloud/floravision-flower-classification"
RELEASE_TAG = "v1.0.0"
MODEL_FILENAME = "floravision_resnet50.keras"
MODEL_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{MODEL_FILENAME}"
MODEL_SHA256 = "48e569fa29f057844b2cee49a96f3c64280c5dcb6c41e6f2df7a838f35c76492"
# The model's output order: tf_flowers label order, NOT alphabetical.
CLASS_NAMES = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]

MAX_DOWNLOAD_BYTES = 300 * 1024 * 1024
MAX_DOWNLOAD_SECONDS = 30 * 60
CHUNK = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_https(url: str) -> None:
    if urlparse(url).scheme != "https":
        raise ValueError(f"Refusing to download over a non-HTTPS URL: {url}")


class _HTTPSOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects (GitHub release assets redirect to a CDN), but only to HTTPS."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HTTPSOnlyRedirects)


def _open(url: str, timeout: float):
    _require_https(url)
    # Every URL, redirects included, is checked to be HTTPS; integrity is checked by the hash.
    return _OPENER.open(url, timeout=timeout)


def download_file(
    url: str,
    dest: str | Path,
    sha256: str,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    timeout: float = 60,
    max_seconds: float = MAX_DOWNLOAD_SECONDS,
) -> Path:
    """Download ``url`` to ``dest`` if its SHA-256 matches; return ``dest``.

    The file is streamed to a temporary file in the same folder and only moved into place
    after the hash check, so ``dest`` never holds a partial or unverified download.
    ``timeout`` limits each network read; ``max_seconds`` limits the whole download.
    """
    _require_https(url)
    dest = Path(dest)
    if dest.is_file() and hmac.compare_digest(sha256_file(dest), sha256.lower()):
        log.info("%s is already present and verified", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name, suffix=".part")
    tmp = Path(tmp_name)
    digest, size = hashlib.sha256(), 0
    deadline = time.monotonic() + max_seconds
    try:
        with os.fdopen(fd, "wb") as out, _open(url, timeout) as resp:
            while block := resp.read(CHUNK):
                size += len(block)
                if size > max_bytes:
                    raise ValueError(f"Download exceeds {max_bytes} bytes; aborting.")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Download took longer than {max_seconds} s; aborting.")
                digest.update(block)
                out.write(block)
        if not hmac.compare_digest(digest.hexdigest(), sha256.lower()):
            raise ValueError(
                f"SHA-256 mismatch for {url}: expected {sha256}, got {digest.hexdigest()}"
            )
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    log.info("Downloaded and verified %s (%.1f MB)", dest, size / 2**20)
    return dest


def download_model(output_dir: str | Path = "models") -> Path:
    """Fetch the published model and write ``class_names.json`` next to it."""
    output_dir = Path(output_dir)
    path = download_file(MODEL_URL, output_dir / MODEL_FILENAME, MODEL_SHA256)
    save_json(CLASS_NAMES, output_dir / "class_names.json")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    download_model(args.output_dir)


if __name__ == "__main__":
    main()
