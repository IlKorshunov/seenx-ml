"""Tests for WPS (words-per-second) feature extraction."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from tests.helpers import load_module


def _install_stubs(monkeypatch, segments: list[dict], duration: float):
    for pkg in ("src", "src.extractors", "src.extractors.text", "src.utils"):
        m = types.ModuleType(pkg)
        m.__path__ = []
        monkeypatch.setitem(sys.modules, pkg, m)

    seenx = types.ModuleType("src.seenx_utils")
    seenx.get_video_duration = lambda _path: duration  # type: ignore[attr-defined]

    log = types.ModuleType("src.utils.logger")
    log.Logger = lambda show=True: types.SimpleNamespace(  # type: ignore[attr-defined]
        get_logger=lambda: types.SimpleNamespace(info=lambda *a, **k: None)
    )

    config = types.ModuleType("src.utils.config")
    config.Config = dict  # type: ignore[attr-defined]

    transcript = types.ModuleType("src.utils.transcript_cache")
    transcript.get_transcript = lambda _path, _cfg: {"segments": segments}  # type: ignore[attr-defined]

    constants = types.ModuleType("src.extractors.text.constants")
    constants.WPS_COLS = {"wps"}  # type: ignore[attr-defined]

    for name, mod in {
        "src.seenx_utils": seenx,
        "src.utils.config": config,
        "src.utils.logger": log,
        "src.utils.transcript_cache": transcript,
        "src.extractors.text.constants": constants,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    load_module("src.extractors.text._base", "src/extractors/text/_base.py")
    return load_module("src.extractors.text.wps_feature", "src/extractors/text/wps_feature.py")


def _segs(*items: tuple[str, float, float]) -> list[dict]:
    return [{"text": t, "start": s, "end": e} for t, s, e in items]


class TestWPSArithmetic:
    def test_uniform_speech_rate(self, monkeypatch):
        wps = _install_stubs(monkeypatch, _segs(("the quick brown fox jumps over the lazy dog", 0.0, 10.0)), 10)
        df = wps.extract_wps("fake.mp4", config=None)
        np.testing.assert_allclose(df["wps"].values, 9 / 10.0, rtol=1e-9)

    def test_two_segments_different_rates(self, monkeypatch):
        segs = _segs(
            ("one two three", 0.0, 5.0),
            ("a b c d e f g h i j k l m n o p", 5.0, 10.0),
        )
        wps = _install_stubs(monkeypatch, segs, 10)
        df = wps.extract_wps("fake.mp4", config=None)
        expected = np.array([0.6] * 5 + [3.2] * 5)
        np.testing.assert_allclose(df["wps"].values, expected, rtol=1e-9)

    def test_silence_gap_at_start_and_end(self, monkeypatch):
        wps = _install_stubs(monkeypatch, _segs(("one two three four five six", 2.0, 8.0)), 10)
        df = wps.extract_wps("fake.mp4", config=None)
        np.testing.assert_allclose(df["wps"].values[:2], 0.0)
        np.testing.assert_allclose(df["wps"].values[8:], 0.0)
        np.testing.assert_allclose(df["wps"].values[2:8], 6 / 6.0, rtol=1e-9)

    def test_zero_duration_segment_is_skipped(self, monkeypatch):
        segs = _segs(("broken", 3.0, 3.0), ("hello world", 0.0, 5.0))
        wps = _install_stubs(monkeypatch, segs, 5)
        df = wps.extract_wps("fake.mp4", config=None)
        assert df["wps"].isna().sum() == 0
        np.testing.assert_allclose(df["wps"].values, 2 / 5.0, rtol=1e-9)

    def test_no_segments_returns_all_zeros(self, monkeypatch):
        wps = _install_stubs(monkeypatch, [], 8)
        df = wps.extract_wps("fake.mp4", config=None)
        assert (df["wps"].values == 0.0).all()

    def test_fast_speech_above_intelligibility_threshold(self, monkeypatch):
        words = " ".join(f"w{i}" for i in range(20))
        wps = _install_stubs(monkeypatch, _segs((words, 0.0, 2.0)), 2)
        df = wps.extract_wps("fake.mp4", config=None)
        np.testing.assert_allclose(df["wps"].values, 10.0, rtol=1e-9)

    def test_segment_end_clamped_to_duration(self, monkeypatch):
        wps = _install_stubs(monkeypatch, _segs(("a b c d e", 8.0, 12.0)), 10)
        df = wps.extract_wps("fake.mp4", config=None)
        assert len(df) == 10
        np.testing.assert_allclose(df["wps"].values[8:10], 5 / 4.0, rtol=1e-9)
        np.testing.assert_allclose(df["wps"].values[:8], 0.0)


class TestWPSOutputContract:
    def test_returns_dataframe_with_wps_column(self, monkeypatch):
        wps = _install_stubs(monkeypatch, [], 5)
        df = wps.extract_wps("fake.mp4", config=None)
        assert isinstance(df, pd.DataFrame)
        assert "wps" in df.columns

    def test_output_length_equals_ceil_of_duration(self, monkeypatch):
        for dur in (1, 5, 60, 300):
            segs = _segs(("hello", 0.0, 2.0))
            wps = _install_stubs(monkeypatch, segs, dur)
            df = wps.extract_wps("fake.mp4", config=None)
            assert len(df) == dur, f"expected {dur} rows, got {len(df)}"

    def test_values_are_non_negative(self, monkeypatch):
        segs = _segs(("hello world", 1.0, 3.0), ("", 4.0, 5.0))
        wps = _install_stubs(monkeypatch, segs, 10)
        df = wps.extract_wps("fake.mp4", config=None)
        assert (df["wps"].values >= 0).all()

    def test_existing_feature_skips_extraction(self, monkeypatch):
        wps = _install_stubs(monkeypatch, [], 5)
        df = wps.extract_wps("fake.mp4", config=None, existing_features=["wps"])
        assert df.empty
