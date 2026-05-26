"""
Tests for WPS (words-per-second) feature extraction.

Strategy: stub out Whisper / get_video_duration so tests are fast and
deterministic, then verify that the aggregation arithmetic in extract_wps
is correct across a range of transcript shapes.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ── module loader (same helper pattern as the rest of the test suite) ──────────

def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_stubs(monkeypatch, segments: list[dict], duration: float):
    """
    Pre-populate sys.modules so that when wps_feature is (re-)loaded it
    picks up lightweight stubs instead of real Whisper / GPU code.

    Returns the freshly loaded wps_feature module.
    """
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

    _load("src.extractors.text._base", "src/extractors/text/_base.py")
    return _load("src.extractors.text.wps_feature", "src/extractors/text/wps_feature.py")


def _segs(*items: tuple[str, float, float]) -> list[dict]:
    return [{"text": t, "start": s, "end": e} for t, s, e in items]


# ── WPS arithmetic ─────────────────────────────────────────────────────────────

class TestWPSArithmetic:

    def test_uniform_speech_rate(self, monkeypatch):
        # 9 words, 10 s → every second should read 0.9
        wps = _install_stubs(monkeypatch, _segs(("the quick brown fox jumps over the lazy dog", 0.0, 10.0)), 10)
        df = wps.extract_wps("fake.mp4", config=None)
        np.testing.assert_allclose(df["wps"].values, 9 / 10.0, rtol=1e-9)

    def test_two_segments_different_rates(self, monkeypatch):
        # 3 words in 0–5 s = 0.6 wps; 16 words in 5–10 s = 3.2 wps
        segs = _segs(
            ("one two three", 0.0, 5.0),
            ("a b c d e f g h i j k l m n o p", 5.0, 10.0),
        )
        wps = _install_stubs(monkeypatch, segs, 10)
        df = wps.extract_wps("fake.mp4", config=None)
        expected = np.array([0.6] * 5 + [3.2] * 5)
        np.testing.assert_allclose(df["wps"].values, expected, rtol=1e-9)

    def test_silence_gap_at_start_and_end(self, monkeypatch):
        # Speech only in seconds 2–8; flanking seconds must be zero
        wps = _install_stubs(monkeypatch, _segs(("one two three four five six", 2.0, 8.0)), 10)
        df = wps.extract_wps("fake.mp4", config=None)
        np.testing.assert_allclose(df["wps"].values[:2], 0.0)
        np.testing.assert_allclose(df["wps"].values[8:], 0.0)
        np.testing.assert_allclose(df["wps"].values[2:8], 6 / 6.0, rtol=1e-9)

    def test_zero_duration_segment_is_skipped(self, monkeypatch):
        # A segment with start == end must not produce NaN / inf
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
        # SPEECH_INTELLIGIBILITY_WPS_FAST = 4.0; exercise the region above it
        words = " ".join(f"w{i}" for i in range(20))
        wps = _install_stubs(monkeypatch, _segs((words, 0.0, 2.0)), 2)
        df = wps.extract_wps("fake.mp4", config=None)
        np.testing.assert_allclose(df["wps"].values, 10.0, rtol=1e-9)

    def test_segment_end_clamped_to_duration(self, monkeypatch):
        # Segment overshoots video end → seg_bounds should clamp to duration
        wps = _install_stubs(monkeypatch, _segs(("a b c d e", 8.0, 12.0)), 10)
        df = wps.extract_wps("fake.mp4", config=None)
        assert len(df) == 10
        # seg_bounds: floor(8.0)=8, min(10, ceil(12.0))=10 → out[8:10] = 5/4.0
        np.testing.assert_allclose(df["wps"].values[8:10], 5 / 4.0, rtol=1e-9)
        np.testing.assert_allclose(df["wps"].values[:8], 0.0)


# ── output contract ────────────────────────────────────────────────────────────

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
        # When "wps" already present in existing_features, returns empty DataFrame
        wps = _install_stubs(monkeypatch, [], 5)
        df = wps.extract_wps("fake.mp4", config=None, existing_features=["wps"])
        assert df.empty
