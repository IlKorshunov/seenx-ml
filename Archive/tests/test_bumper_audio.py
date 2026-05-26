"""
Unit tests for src/cutting_shots/audio.py.

Covers the pure-NumPy signal-processing functions:
  slide_score, locate_in_video, find_bumper_template

All tests work on synthetic chroma arrays — no real audio files needed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ── loader / stubs ─────────────────────────────────────────────────────────────

def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_stubs(monkeypatch):
    for pkg in ("src", "src.cutting_shots", "src.utils"):
        m = types.ModuleType(pkg)
        m.__path__ = []
        monkeypatch.setitem(sys.modules, pkg, m)

    seenx = types.ModuleType("src.seenx_utils")
    seenx.get_video_duration = lambda _p: 60.0  # type: ignore[attr-defined]

    log = types.ModuleType("src.utils.logger")
    log.Logger = lambda show=True: types.SimpleNamespace(  # type: ignore[attr-defined]
        get_logger=lambda: types.SimpleNamespace(
            info=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
    )

    for name, mod in {"src.seenx_utils": seenx, "src.utils.logger": log}.items():
        monkeypatch.setitem(sys.modules, name, mod)

    cfg_mod = _load("src.cutting_shots.configs", "src/cutting_shots/configs.py")
    audio_mod = _load("src.cutting_shots.audio", "src/cutting_shots/audio.py")
    return audio_mod, cfg_mod


# ── test helpers ───────────────────────────────────────────────────────────────

def _unit_chroma(n_frames: int, seed: int = 0) -> np.ndarray:
    """Random L2-normalised chroma matrix (n_frames × 12)."""
    rng = np.random.default_rng(seed)
    c = rng.random((n_frames, 12)).astype(np.float64) + 0.1
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)


def _embed(base: np.ndarray, template: np.ndarray, pos: int) -> np.ndarray:
    """Return a copy of base with template inserted at frame position pos."""
    out = base.copy()
    out[pos: pos + len(template)] = template
    return out


def _cfg(cfg_mod, **overrides):
    defaults = dict(
        audio_sr=22050,
        chroma_fps=2,
        scan_ratio=0.4,
        min_bumper_sec=2,
        max_bumper_sec=5,
        match_threshold=0.80,
        mask_score_ratio=0.9,
        mask_radius_sec=2,
        silence_thresh=0.005,
        locate_threshold=0.75,
        locate_max_per_video=3,
        min_candidate_videos=2,
    )
    defaults.update(overrides)
    return cfg_mod.BumperConfig(**defaults)


# ── slide_score ────────────────────────────────────────────────────────────────

class TestSlideScore:

    def test_self_match_scores_near_one(self, monkeypatch):
        audio, _ = _install_stubs(monkeypatch)
        chroma = _unit_chroma(50)
        ref = chroma[10:16]
        scores = audio.slide_score(ref, chroma)
        assert scores.max() == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_score_zero(self, monkeypatch):
        audio, _ = _install_stubs(monkeypatch)
        ref = np.zeros((5, 12))
        ref[:, 0] = 1.0        # only first chroma bin
        target = np.zeros((20, 12))
        target[:, 6] = 1.0     # only seventh bin → orthogonal
        scores = audio.slide_score(ref, target)
        assert scores.max() == pytest.approx(0.0, abs=1e-6)

    def test_output_length(self, monkeypatch):
        audio, _ = _install_stubs(monkeypatch)
        ref = _unit_chroma(5)
        target = _unit_chroma(30)
        scores = audio.slide_score(ref, target)
        assert len(scores) == 30 - 5 + 1

    def test_target_shorter_than_ref_returns_empty(self, monkeypatch):
        audio, _ = _install_stubs(monkeypatch)
        ref = _unit_chroma(10)
        target = _unit_chroma(5)
        scores = audio.slide_score(ref, target)
        assert len(scores) == 0

    def test_scores_bounded_minus_one_to_one(self, monkeypatch):
        audio, _ = _install_stubs(monkeypatch)
        ref = _unit_chroma(6, seed=77)
        target = _unit_chroma(40, seed=88)
        scores = audio.slide_score(ref, target)
        assert scores.min() >= -1.0 - 1e-6
        assert scores.max() <= 1.0 + 1e-6


# ── locate_in_video ────────────────────────────────────────────────────────────

class TestLocateInVideo:

    def test_finds_embedded_template(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod)
        base = _unit_chroma(100)
        template = _unit_chroma(8, seed=99)
        pos = 30
        chroma = _embed(base, template, pos)

        matches = audio.locate_in_video(template, chroma, cfg)

        assert len(matches) >= 1
        found_sec, score = matches[0]
        expected_sec = pos / cfg.chroma_fps
        assert found_sec == pytest.approx(expected_sec, abs=1 / cfg.chroma_fps)
        assert score >= cfg.locate_threshold

    def test_no_match_below_threshold(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, locate_threshold=0.99)
        template = _unit_chroma(8, seed=1)
        noise = _unit_chroma(100, seed=2)   # independent random → low similarity
        matches = audio.locate_in_video(template, noise, cfg)
        assert len(matches) == 0

    def test_respects_locate_max_per_video(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, locate_max_per_video=2)
        template = _unit_chroma(4, seed=5)
        chroma = _unit_chroma(200, seed=7)
        # Embed the same template at three well-separated positions
        for pos in (10, 70, 140):
            chroma = _embed(chroma, template, pos)
        matches = audio.locate_in_video(template, chroma, cfg)
        assert len(matches) <= 2

    def test_returned_scores_non_increasing(self, monkeypatch):
        # Scores should come back in descending order (best match first)
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, locate_max_per_video=3)
        template = _unit_chroma(6, seed=42)
        chroma = _unit_chroma(120, seed=11)
        chroma = _embed(chroma, template, 20)
        chroma = _embed(chroma, template, 80)
        matches = audio.locate_in_video(template, chroma, cfg)
        scores = [m[1] for m in matches]
        assert scores == sorted(scores, reverse=True)


# ── find_bumper_template ───────────────────────────────────────────────────────

class TestFindBumperTemplate:

    def test_detects_shared_segment(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, min_bumper_sec=2, max_bumper_sec=4, match_threshold=0.80)
        fps = cfg.chroma_fps
        template_len = fps * 3           # 3-second template @ 2 fps = 6 frames
        template = _unit_chroma(template_len, seed=100)

        chroma_a = _embed(_unit_chroma(60, seed=1), template, 5)
        chroma_b = _embed(_unit_chroma(60, seed=2), template, 20)

        result = audio.find_bumper_template(chroma_a, chroma_b, cfg)

        assert result is not None, "Expected to find a matching bumper template"
        _tmpl, sa, sb, dur, score = result
        assert score >= cfg.match_threshold
        assert dur >= cfg.min_bumper_sec

    def test_returns_none_when_no_match(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, match_threshold=0.999)   # impossibly strict
        chroma_a = _unit_chroma(40, seed=3)
        chroma_b = _unit_chroma(40, seed=4)
        result = audio.find_bumper_template(chroma_a, chroma_b, cfg)
        assert result is None

    def test_mask_prevents_reuse_of_same_region(self, monkeypatch):
        # With a fully-set mask over the template region in chroma_a,
        # the same match should not be returned.
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, min_bumper_sec=2, max_bumper_sec=4, match_threshold=0.80)
        fps = cfg.chroma_fps
        template = _unit_chroma(fps * 3, seed=55)
        chroma_a = _embed(_unit_chroma(60, seed=10), template, 4)
        chroma_b = _embed(_unit_chroma(60, seed=11), template, 10)

        # Mask the entire chroma_a so nothing can be chosen from it
        mask_a = np.ones(chroma_a.shape[0], dtype=bool)
        result = audio.find_bumper_template(chroma_a, chroma_b, cfg, mask_a=mask_a)
        assert result is None
