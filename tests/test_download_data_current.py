import os
import sys
import types
import json
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


class _Tqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def update(self, *_args, **_kwargs):
        return None

    def close(self):
        return None

    @staticmethod
    def write(*_args, **_kwargs):
        return None


tqdm_stub.tqdm = _Tqdm  # type: ignore[attr-defined]
sys.modules.setdefault("tqdm", tqdm_stub)
auto_stub = types.ModuleType("tqdm.auto")
auto_stub.__spec__ = ModuleSpec("tqdm.auto", loader=None)
auto_stub.tqdm = _Tqdm  # type: ignore[attr-defined]
sys.modules.setdefault("tqdm.auto", auto_stub)
contrib_stub = types.ModuleType("tqdm.contrib")
contrib_stub.__spec__ = ModuleSpec("tqdm.contrib", loader=None, is_package=True)
concurrent_stub = types.ModuleType("tqdm.contrib.concurrent")
concurrent_stub.__spec__ = ModuleSpec("tqdm.contrib.concurrent", loader=None)
concurrent_stub.thread_map = lambda fn, iterable, *a, **k: [fn(item) for item in iterable]  # type: ignore[attr-defined]
sys.modules.setdefault("tqdm.contrib", contrib_stub)
sys.modules.setdefault("tqdm.contrib.concurrent", concurrent_stub)
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


def test_api_get_builds_url_and_decodes_json(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(url, timeout):
        seen.update({"url": url, "timeout": timeout})
        return Resp()

    monkeypatch.setattr(download_data, "urlopen", fake_urlopen)

    assert download_data._api_get({"q": "a b", "key": "KEY"}) == {"ok": True}
    assert "q=a+b" in seen["url"]
    assert seen["timeout"] == 60


def test_fetch_file_index_api_collects_folder_files(monkeypatch):
    def fake_iter(query, fields, api_key):
        assert api_key == "KEY"
        if "mimeType='application/vnd.google-apps.folder'" in query:
            return iter([{"id": "folder-id", "name": "folder"}])
        return iter([{"id": "file-id", "name": "video.mp4"}])

    monkeypatch.setattr(download_data, "_api_iter_files", fake_iter)

    assert download_data.fetch_file_index_api("KEY") == OrderedDict({"folder": [("file-id", "video.mp4")]})


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


def test_low_level_download_helpers(monkeypatch, tmp_path):
    gdown_out = tmp_path / "gdown.bin"

    def fake_gdown_download(id, output, quiet):
        assert id == "gid"
        assert quiet is True
        with open(output, "wb") as f:
            f.write(b"x" * 120)

    monkeypatch.setattr(download_data.gdown, "download", fake_gdown_download)
    assert download_data._download_single_file_gdown("gid", str(gdown_out)) is True

    class Resp:
        def __init__(self):
            self.chunks = [b"a" * 80, b"b" * 80, b""]

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self, _n):
            return self.chunks.pop(0)

    seen = {}

    def fake_urlopen(url, timeout):
        seen.update({"url": url, "timeout": timeout})
        return Resp()

    api_out = tmp_path / "api.bin"
    monkeypatch.setattr(download_data, "urlopen", fake_urlopen)
    assert download_data._download_single_file_api("file id", str(api_out), "K EY") is True
    assert "file%20id" in seen["url"]
    assert "K%20EY" in seen["url"]
    assert api_out.stat().st_size == 160


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


def test_download_single_file_exhausts_retries_and_reports_last_error(monkeypatch, tmp_path):
    out = tmp_path / "bad.bin"
    writes = []

    def fake_download(*_args):
        writes.append("try")
        if len(writes) == 1:
            out.write_bytes(b"x" * 50)
            return False
        raise RuntimeError("network")

    messages = []
    monkeypatch.setattr(download_data, "_download_single_file_gdown", fake_download)
    monkeypatch.setattr(download_data.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(download_data.tqdm, "write", lambda msg: messages.append(msg))
    monkeypatch.setattr(download_data, "MAX_RETRIES", 2)

    assert download_data.download_single_file("id", str(out), method="gdown", api_key=None) is False
    assert not out.exists()
    assert any("Error: network" in msg for msg in messages)


def test_download_folder_removes_small_existing_file(monkeypatch, tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    small = folder / "small.bin"
    small.write_bytes(b"x")
    calls = []

    def fake_download(_file_id, output_path, *_args):
        calls.append(os.path.basename(output_path))
        with open(output_path, "wb") as f:
            f.write(b"x" * 120)
        return True

    monkeypatch.setattr(download_data, "download_single_file", fake_download)
    monkeypatch.setattr(download_data.time, "sleep", lambda *_a, **_k: None)

    assert download_data.download_folder_files("folder", [("id", "small.bin")], str(tmp_path), "gdown", None) is True
    assert calls == ["small.bin"]
    assert small.stat().st_size == 120


def test_main_branches_downloads_and_early_returns(monkeypatch, tmp_path, capsys):
    data_dir = tmp_path / "data"
    (data_dir / "existing").mkdir(parents=True)
    remote = OrderedDict({"existing": [("1", "a")], "new": [("2", "b")]})
    downloaded = []

    monkeypatch.setattr(sys, "argv", ["download_data.py", "--count", "5", "--data_dir", str(data_dir), "--method", "gdown"])
    monkeypatch.setattr(download_data, "fetch_file_index", lambda method, api_key: remote)
    monkeypatch.setattr(download_data, "download_folder_files", lambda name, files, *_args: downloaded.append((name, files)) or False)

    download_data.main()
    text = capsys.readouterr().out
    assert "Warning: requested 5" in text
    assert downloaded == [("new", [("2", "b")])]

    downloaded.clear()
    monkeypatch.setattr(sys, "argv", ["download_data.py", "--count", "1", "--data_dir", str(data_dir)])
    download_data.main()
    assert downloaded == []
