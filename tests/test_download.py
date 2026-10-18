"""The model downloader must only ever keep a file whose SHA-256 matches."""

import hashlib
import io
import json

import pytest

from floravision import download

PAYLOAD = b"pretend this is a .keras model" * 1000
GOOD_HASH = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def fake_urlopen(monkeypatch):
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        return FakeResponse(PAYLOAD)

    monkeypatch.setattr(download._OPENER, "open", urlopen)
    return calls


def test_verified_download_is_written(tmp_path, fake_urlopen):
    dest = download.download_file("https://example.org/m.keras", tmp_path / "m.keras", GOOD_HASH)
    assert dest.read_bytes() == PAYLOAD
    assert list(tmp_path.iterdir()) == [dest]  # no leftover .part file


def test_existing_verified_file_is_not_downloaded_again(tmp_path, fake_urlopen):
    (tmp_path / "m.keras").write_bytes(PAYLOAD)
    download.download_file("https://example.org/m.keras", tmp_path / "m.keras", GOOD_HASH)
    assert fake_urlopen == []


def test_hash_mismatch_leaves_nothing_behind(tmp_path, fake_urlopen):
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download.download_file("https://example.org/m.keras", tmp_path / "m.keras", "0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_tampered_existing_file_is_replaced(tmp_path, fake_urlopen):
    (tmp_path / "m.keras").write_bytes(b"tampered")
    download.download_file("https://example.org/m.keras", tmp_path / "m.keras", GOOD_HASH)
    assert (tmp_path / "m.keras").read_bytes() == PAYLOAD


@pytest.mark.parametrize(
    "url", ["http://example.org/m.keras", "file:///etc/passwd", "ftp://example.org/m.keras"]
)
def test_non_https_urls_are_refused(tmp_path, fake_urlopen, url):
    with pytest.raises(ValueError, match="non-HTTPS"):
        download.download_file(url, tmp_path / "m.keras", GOOD_HASH)
    assert fake_urlopen == []


def test_download_size_is_capped(tmp_path, fake_urlopen):
    with pytest.raises(ValueError, match="exceeds"):
        download.download_file(
            "https://example.org/m.keras", tmp_path / "m.keras", GOOD_HASH, max_bytes=100
        )
    assert list(tmp_path.iterdir()) == []


def test_download_model_writes_class_names_in_model_order(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "download_file", lambda url, dest, sha: dest)
    download.download_model(tmp_path)
    names = json.loads((tmp_path / "class_names.json").read_text())
    assert names == ["dandelion", "daisy", "tulips", "sunflowers", "roses"]


def test_published_settings_are_pinned():
    assert download.MODEL_URL.startswith("https://github.com/")
    assert len(download.MODEL_SHA256) == 64
    assert set(download.MODEL_SHA256) <= set("0123456789abcdef")


@pytest.mark.parametrize("target", ["http://cdn.example.org/m.keras", "ftp://example.org/m"])
def test_redirects_to_non_https_are_refused(target):
    import urllib.request

    handler = download._HTTPSOnlyRedirects()
    req = urllib.request.Request("https://github.com/x/releases/download/v1/m.keras")
    with pytest.raises(ValueError, match="non-HTTPS"):
        handler.redirect_request(req, None, 302, "Found", {}, target)


def test_https_redirects_are_followed():
    import urllib.request

    handler = download._HTTPSOnlyRedirects()
    req = urllib.request.Request("https://github.com/x/releases/download/v1/m.keras")
    new = handler.redirect_request(req, None, 302, "Found", {}, "https://cdn.example.org/m")
    assert new.full_url == "https://cdn.example.org/m"


def test_slow_downloads_are_aborted(tmp_path, fake_urlopen):
    with pytest.raises(TimeoutError):
        download.download_file(
            "https://example.org/m.keras", tmp_path / "m.keras", GOOD_HASH, max_seconds=-1
        )
    assert list(tmp_path.iterdir()) == []
