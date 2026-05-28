"""Unit tests for src/cutting_shots/audio.py."""

import sys
import types

import numpy as np
import pytest

from tests.helpers import load_module


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

    cfg_mod = load_module("src.cutting_shots.configs", "src/cutting_shots/configs.py")
    audio_mod = load_module("src.cutting_shots.audio", "src/cutting_shots/audio.py")
    return audio_mod, cfg_mod


def _unit_chroma(n_frames: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.random((n_frames, 12)).astype(np.float64) + 0.1
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)


def _embed(base: np.ndarray, template: np.ndarray, pos: int) -> np.ndarray:
    out = base.copy()
    out[pos : pos + len(template)] = template
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
        ref[:, 0] = 1.0
        target = np.zeros((20, 12))
        target[:, 6] = 1.0
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
        noise = _unit_chroma(100, seed=2)
        matches = audio.locate_in_video(template, noise, cfg)
        assert len(matches) == 0

    def test_respects_locate_max_per_video(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, locate_max_per_video=2)
        template = _unit_chroma(4, seed=5)
        chroma = _unit_chroma(200, seed=7)
        for pos in (10, 70, 140):
            chroma = _embed(chroma, template, pos)
        matches = audio.locate_in_video(template, chroma, cfg)
        assert len(matches) <= 2

    def test_returned_scores_non_increasing(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, locate_max_per_video=3)
        template = _unit_chroma(6, seed=42)
        chroma = _unit_chroma(120, seed=11)
        chroma = _embed(chroma, template, 20)
        chroma = _embed(chroma, template, 80)
        matches = audio.locate_in_video(template, chroma, cfg)
        scores = [m[1] for m in matches]
        assert scores == sorted(scores, reverse=True)


class TestFindBumperTemplate:
    def test_detects_shared_segment(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, min_bumper_sec=2, max_bumper_sec=4, match_threshold=0.80)
        fps = cfg.chroma_fps
        template_len = fps * 3
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
        cfg = _cfg(cfg_mod, match_threshold=0.999)
        chroma_a = _unit_chroma(40, seed=3)
        chroma_b = _unit_chroma(40, seed=4)
        result = audio.find_bumper_template(chroma_a, chroma_b, cfg)
        assert result is None

    def test_mask_prevents_reuse_of_same_region(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, min_bumper_sec=2, max_bumper_sec=4, match_threshold=0.80)
        fps = cfg.chroma_fps
        template = _unit_chroma(fps * 3, seed=55)
        chroma_a = _embed(_unit_chroma(60, seed=10), template, 4)
        chroma_b = _embed(_unit_chroma(60, seed=11), template, 10)

        mask_a = np.ones(chroma_a.shape[0], dtype=bool)
        result = audio.find_bumper_template(chroma_a, chroma_b, cfg, mask_a=mask_a)
        assert result is None


class TestExtractChroma:
    def test_extract_chroma_normalizes_pads_rms_and_mutes_silence(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, audio_sr=8, chroma_fps=2, silence_thresh=0.2)
        y = np.ones(16, dtype=np.float32)
        rms = np.array([[0.1], [0.5]], dtype=np.float32)
        chroma = np.array(
            [
                [3.0, 4.0] + [0.0] * 10,
                [1.0, 0.0] + [0.0] * 10,
                [0.0, 2.0] + [0.0] * 10,
            ],
            dtype=np.float32,
        )

        monkeypatch.setattr(audio.librosa, "load", lambda path, sr, mono, duration: (y, sr))
        monkeypatch.setattr(audio.librosa.feature, "rms", lambda **_kwargs: rms.T)
        monkeypatch.setattr(audio.librosa.feature, "chroma_cqt", lambda **_kwargs: chroma.T)

        out = audio.extract_chroma("fake.mp4", cfg, max_sec=3.0)

        assert out.shape == (3, 12)
        np.testing.assert_allclose(out[0], np.zeros(12), atol=1e-6)
        assert np.linalg.norm(out[1]) == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(out[2], np.zeros(12), atol=1e-6)

    def test_extract_chroma_truncates_longer_rms(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, audio_sr=8, chroma_fps=2, silence_thresh=0.2)
        chroma = np.ones((2, 12), dtype=np.float32)
        rms = np.array([[0.1], [0.5], [0.1], [0.5]], dtype=np.float32)

        monkeypatch.setattr(audio.librosa, "load", lambda path, sr, mono, duration: (np.ones(16), sr))
        monkeypatch.setattr(audio.librosa.feature, "rms", lambda **_kwargs: rms.T)
        monkeypatch.setattr(audio.librosa.feature, "chroma_cqt", lambda **_kwargs: chroma.T)

        out = audio.extract_chroma("fake.mp4", cfg)

        assert out.shape == (2, 12)
        np.testing.assert_allclose(out[0], np.zeros(12), atol=1e-6)
        assert out[1].sum() > 0


class TestFindAllTemplates:
    def test_finds_template_across_video_pairs_and_masks_reuse(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, min_bumper_sec=2, max_bumper_sec=3, match_threshold=0.80, mask_score_ratio=0.9, mask_radius_sec=1)
        fps = cfg.chroma_fps
        template = _unit_chroma(fps * 2, seed=123)
        chromas = {
            "a": _embed(_unit_chroma(70, seed=1), template, 4),
            "b": _embed(_unit_chroma(70, seed=2), template, 18),
            "c": _embed(_unit_chroma(70, seed=3), template, 32),
        }

        templates = audio.find_all_templates(chromas, ["a", "b", "c"], cfg, max_types=2)

        assert templates
        found_template, dur_sec, score, pair = templates[0]
        assert found_template.shape[0] == dur_sec * fps
        assert score >= cfg.match_threshold
        assert set(pair) <= {"a", "b", "c"}

    def test_returns_empty_when_no_pair_matches(self, monkeypatch):
        audio, cfg_mod = _install_stubs(monkeypatch)
        cfg = _cfg(cfg_mod, match_threshold=1.1)
        chromas = {"a": _unit_chroma(30, seed=1), "b": _unit_chroma(30, seed=2)}

        assert audio.find_all_templates(chromas, ["a", "b"], cfg, max_types=1) == []


class TestRunAudioPipeline:
    def test_requires_at_least_two_videos(self, monkeypatch, tmp_path):
        audio, cfg_mod = _install_stubs(monkeypatch)
        (tmp_path / "one").mkdir()
        (tmp_path / "one" / "video.mp4").write_text("", encoding="utf-8")

        candidates, paths = audio.run_audio_pipeline(str(tmp_path), _cfg(cfg_mod))

        assert candidates == []
        assert paths == {}

    def test_returns_paths_when_no_templates_found(self, monkeypatch, tmp_path):
        audio, cfg_mod = _install_stubs(monkeypatch)
        for vid in ("a", "b"):
            (tmp_path / vid).mkdir()
            (tmp_path / vid / "video.mp4").write_text("", encoding="utf-8")
        cfg = _cfg(cfg_mod)

        monkeypatch.setattr(audio, "extract_chroma", lambda path, cfg, max_sec=None: _unit_chroma(20, seed=len(path)))
        monkeypatch.setattr(audio, "find_all_templates", lambda *_a, **_k: [])

        candidates, paths = audio.run_audio_pipeline(str(tmp_path), cfg)

        assert candidates == []
        assert set(paths) == {"a", "b"}

    def test_builds_candidates_and_filters_types_by_video_count(self, monkeypatch, tmp_path):
        audio, cfg_mod = _install_stubs(monkeypatch)
        for vid in ("a", "b", "c"):
            (tmp_path / vid).mkdir()
            (tmp_path / vid / "video.mp4").write_text("", encoding="utf-8")
        cfg = _cfg(cfg_mod, min_candidate_videos=2)
        template0 = _unit_chroma(4, seed=44)
        template1 = _unit_chroma(4, seed=55)

        monkeypatch.setattr(audio, "extract_chroma", lambda path, cfg, max_sec=None: _unit_chroma(30, seed=len(path)))
        monkeypatch.setattr(audio, "find_all_templates", lambda *_a, **_k: [(template0, 2, 0.95, ("a", "b")), (template1, 2, 0.94, ("a", "c"))])

        def fake_locate(template, chroma, cfg):
            if template is template0:
                return [(1.234, 0.923)]
            return [(5.0, 0.8)] if float(chroma[0, 0]) > 0.0 else []

        calls = {"n": 0}

        def locate_with_filter(template, chroma, cfg):
            calls["n"] += 1
            if template is template0:
                return [(1.234, 0.923)]
            return [(5.0, 0.8)] if calls["n"] == 4 else []

        monkeypatch.setattr(audio, "locate_in_video", locate_with_filter)

        candidates, paths = audio.run_audio_pipeline(str(tmp_path), cfg, max_types=2)

        assert set(paths) == {"a", "b", "c"}
        assert len(candidates) == 3
        assert {candidate.bumper_type for candidate in candidates} == {0}
        assert candidates[0].start_sec == 1.23
        assert candidates[0].end_sec == 3.23
        assert candidates[0].audio_score == 0.923
