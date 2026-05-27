import os
import sys
import types
from importlib.machinery import ModuleSpec
from collections import OrderedDict

import pytest  # type: ignore[reportMissingImports]

from tests.helpers import load_module


gdown_stub = types.ModuleType("gdown")
gdown_stub.download_folder = lambda *a, **k: []  # type: ignore[attr-defined]
gdown_stub.download = lambda *a, **k: None  # type: ignore[attr-defined]
sys.modules.setdefault("gdown", gdown_stub)

tqdm_stub = types.ModuleType("tqdm")
tqdm_stub.__path__ = []  # type: ignore[attr-defined]
tqdm_stub.__spec__ = ModuleSpec("tqdm", loader=None, is_package=True)


def _tqdm(iterable=None, *args, **kwargs):
    return iterable if iterable is not None else []


_tqdm.write = lambda *a, **k: None  # type: ignore[attr-defined]
tqdm_stub.tqdm = _tqdm  # type: ignore[attr-defined]
sys.modules.setdefault("tqdm", tqdm_stub)
auto_stub = types.ModuleType("tqdm.auto")
auto_stub.__spec__ = ModuleSpec("tqdm.auto", loader=None)
auto_stub.tqdm = _tqdm  # type: ignore[attr-defined]
sys.modules.setdefault("tqdm.auto", auto_stub)
download_data = load_module("download_data", "other/download_data.py")


def test_fetch_file_index_gdown_parses_nested_paths(monkeypatch):
    files = [
        types.SimpleNamespace(id="1", path="folder_a/video.mp4"),
        types.SimpleNamespace(id="2", path="folder_a/meta.json"),
        types.SimpleNamespace(id="3", path="README.md"),
        types.SimpleNamespace(id="4", path="folder_b/audio.mp3"),
    ]
    monkeypatch.setattr(download_data.gdown, "download_folder", lambda *a, **k: files)

    result = download_data.fetch_file_index_gdown()

    assert isinstance(result, OrderedDict)
    assert list(result) == ["folder_a", "folder_b"]
    assert result["folder_a"] == [("1", "video.mp4"), ("2", "meta.json")]
    assert result["folder_b"] == [("4", "audio.mp3")]


def test_fetch_file_index_dispatch_and_api_key_validation(monkeypatch):
    monkeypatch.setattr(download_data, "fetch_file_index_gdown", lambda: OrderedDict({"g": []}))
    monkeypatch.setattr(download_data, "fetch_file_index_api", lambda api_key: OrderedDict({api_key: []}))

    assert list(download_data.fetch_file_index("gdown", None)) == ["g"]
    assert list(download_data.fetch_file_index("api", "KEY")) == ["KEY"]
    with pytest.raises(ValueError):
        download_data.fetch_file_index("api", None)


def test_api_iter_files_handles_pagination(monkeypatch):
    calls = []

    def fake_api_get(params):
        calls.append(params)
        if "pageToken" not in params:
            return {"files": [{"id": "1"}], "nextPageToken": "next"}
        return {"files": [{"id": "2"}]}

    monkeypatch.setattr(download_data, "_api_get", fake_api_get)

    assert list(download_data._api_iter_files("q", "id", "KEY")) == [{"id": "1"}, {"id": "2"}]
    assert calls[1]["pageToken"] == "next"


def test_download_single_file_retries_and_removes_small_files(monkeypatch, tmp_path):
    out = tmp_path / "file.bin"
    attempts = {"n": 0}

    def fake_download(file_id, output_path):
        attempts["n"] += 1
        out.write_bytes(b"x" * (50 if attempts["n"] == 1 else 200))
        return out.stat().st_size > 100

    monkeypatch.setattr(download_data, "_download_single_file_gdown", fake_download)
    monkeypatch.setattr(download_data.time, "sleep", lambda *_a, **_k: None)

    assert download_data.download_single_file("id", str(out), method="gdown", api_key=None) is True
    assert attempts["n"] == 2
    assert out.stat().st_size == 200


def test_download_single_file_api_backend(monkeypatch, tmp_path):
    out = tmp_path / "file.bin"
    seen = {}

    def fake_download(file_id, output_path, api_key):
        seen.update({"file_id": file_id, "output_path": output_path, "api_key": api_key})
        out.write_bytes(b"x" * 200)
        return True

    monkeypatch.setattr(download_data, "_download_single_file_api", fake_download)

    assert download_data.download_single_file("id", str(out), method="api", api_key="KEY") is True
    assert seen == {"file_id": "id", "output_path": str(out), "api_key": "KEY"}


def test_download_folder_files_skips_existing_and_reports_failures(monkeypatch, tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "existing.bin").write_bytes(b"x" * 200)
    calls = []

    def fake_download(file_id, output_path, method, api_key):
        calls.append((file_id, os.path.basename(output_path), method, api_key))
        if file_id == "bad":
            return False
        with open(output_path, "wb") as f:
            f.write(b"x" * 200)
        return True

    monkeypatch.setattr(download_data, "download_single_file", fake_download)
    monkeypatch.setattr(download_data.time, "sleep", lambda *_a, **_k: None)

    ok = download_data.download_folder_files(
        "folder",
        [("skip", "existing.bin"), ("good", "new.bin"), ("bad", "bad.bin")],
        str(tmp_path),
        method="api",
        api_key="KEY",
    )

    assert ok is False
    assert calls == [("good", "new.bin", "api", "KEY"), ("bad", "bad.bin", "api", "KEY")]
