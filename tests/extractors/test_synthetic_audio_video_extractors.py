"""Synthetic coverage for lightweight audio/video extractors."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from tests.helpers import load_module


def _install_video_stubs(monkeypatch):
    for package_name in ("src", "src.extractors", "src.extractors.video", "src.utils"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    config = types.ModuleType("src.utils.config")
    config.Config = dict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.utils.config", config)

    feature_extractor = types.ModuleType("src.extractors.feature_extractor")
    feature_extractor.VideoFeature = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.extractors.feature_extractor", feature_extractor)

    load_module("src.extractors.video.constants", "src/extractors/video/constants.py")
    return load_module("src.extractors.video.common", "src/extractors/video/common.py")


def _install_audio_stubs(monkeypatch):
    for package_name in ("src", "src.extractors", "src.extractors.audio", "src.utils"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    config = types.ModuleType("src.utils.config")
    config.Config = dict  # type: ignore[attr-defined]

    logger = types.ModuleType("src.utils.logger")
    logger.Logger = lambda show=True: types.SimpleNamespace(  # type: ignore[attr-defined]
        get_logger=lambda: types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    )

    seenx = types.ModuleType("src.seenx_utils")
    seenx.get_video_duration = lambda _path: 6.0  # type: ignore[attr-defined]

    audio_utils = types.ModuleType("src.audio_utils")
    audio_utils.extract_audio_to_wav = lambda _path, sr: "tmp.wav"  # type: ignore[attr-defined]

    for module_name, module in {
        "src.utils.config": config,
        "src.utils.logger": logger,
        "src.seenx_utils": seenx,
        "src.audio_utils": audio_utils,
    }.items():
        monkeypatch.setitem(sys.modules, module_name, module)

    load_module("src.extractors.audio.consts", "src/extractors/audio/consts.py")
    return load_module("src.extractors.audio.common", "src/extractors/audio/common.py")


class FakeCapture:
    def __init__(self, frames, fps=2.0):
        self.frames = list(frames)
        self.fps = fps
        self.idx = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def read(self):
        if self.idx >= len(self.frames):
            return False, None
        frame = self.frames[self.idx]
        self.idx += 1
        return True, frame.copy()

    def release(self):
        self.released = True


class FakeCaptureContext:
    def __init__(self, frames):
        self.frames = frames

    def __enter__(self):
        return FakeCapture(self.frames)

    def __exit__(self, *_exc):
        return None


def _frame(bgr):
    return np.full((4, 4, 3), bgr, dtype=np.uint8)


class TestVideoCommon:
    def test_video_id_embeddings_and_mask_runs(self, monkeypatch, tmp_path):
        common = _install_video_stubs(monkeypatch)

        assert common.video_id("/data/abc/video.mp4") == "abc"
        assert common.video_id("/data/custom.mov") == "custom"
        assert common.embeddings_dir("/data/abc/video.mp4", embeddings_root=tmp_path) == tmp_path / "abc"
        np.testing.assert_array_equal(common.mask_runs(np.array([0, 1, 1, 0, 1], dtype=bool)), np.array([[1, 2], [4, 4]], dtype=np.int32))

    def test_frame_iteration_helpers_use_capture_fps_and_stride(self, monkeypatch):
        common = _install_video_stubs(monkeypatch)
        frames = [_frame((i, i, i)) for i in range(5)]
        monkeypatch.setattr(common.cv2, "VideoCapture", lambda _path: FakeCapture(frames, fps=2.0))

        assert common.read_video_fps("video.mp4") == 2.0
        assert len(list(common.iter_video_frames("video.mp4"))) == 5
        assert len(common.iter_1fps_rgb_frames("video.mp4")) == 3


class TestVideoFeatures:
    def test_edit_pace_counts_cuts_per_minute(self, monkeypatch):
        _install_video_stubs(monkeypatch)
        module = load_module("src.extractors.video.edit_pace_feature", "src/extractors/video/edit_pace_feature.py")
        monkeypatch.setattr(module, "get_video_fps", lambda _path: 1.0)
        monkeypatch.setattr(module, "EDIT_PACE_WINDOW_SEC", 4)
        monkeypatch.setattr(module, "EDIT_PACE_MIN_WINDOW_SEC", 1)
        df = pd.DataFrame(index=range(6))

        module.EditPaceFeature(config={}).run("video.mp4", {"data": df, "shot_bounds": [0, 1, 2, 4, 5, 5]})

        assert "edit_pace" in df
        assert df["edit_pace"].max() > 0

    def test_short_insert_marks_short_shots_and_rolling_rate(self, monkeypatch):
        _install_video_stubs(monkeypatch)
        module = load_module("src.extractors.video.short_insert_feature", "src/extractors/video/short_insert_feature.py")
        monkeypatch.setattr(module, "get_video_fps", lambda _path: 1.0)
        module.ShortInsertFeature.RATE_WINDOW_SEC = 3
        module.ShortInsertFeature.SHORT_THRESHOLD_SEC = 2
        df = pd.DataFrame(index=range(8))

        module.ShortInsertFeature(config={}).run("video.mp4", {"data": df, "shot_bounds": [0, 1, 2, 6]})

        np.testing.assert_allclose(df["short_insert"].values[:2], [1.0, 1.0])
        assert df["short_insert_rate"].between(0, 1).all()

    def test_color_feature_writes_temperature_and_saturation(self, monkeypatch):
        _install_video_stubs(monkeypatch)
        module = load_module("src.extractors.video.color_features", "src/extractors/video/color_features.py")
        monkeypatch.setattr(module, "iter_video_frames", lambda _path: iter([_frame((10, 20, 200)), _frame((200, 20, 10))]))
        df = pd.DataFrame(index=range(3))

        module.ColorFeature(config={}).run("video.mp4", {"data": df})

        assert df["color_temperature"].iloc[0] > 0
        assert df["color_temperature"].iloc[1] < 0
        assert pd.isna(df["color_temperature"].iloc[2])

    def test_visual_entropy_and_frame_quality_use_fake_capture(self, monkeypatch):
        common = _install_video_stubs(monkeypatch)
        frames = [_frame((0, 0, 0)), _frame((255, 255, 255)), np.dstack([np.tile(np.arange(4, dtype=np.uint8), (4, 1))] * 3)]
        monkeypatch.setattr(common, "open_video_capture", lambda _path: FakeCaptureContext(frames))

        entropy_module = load_module("src.extractors.video.visual_entropy_feature", "src/extractors/video/visual_entropy_feature.py")
        quality_module = load_module("src.extractors.video.frame_feature", "src/extractors/video/frame_feature.py")
        monkeypatch.setattr(entropy_module, "open_video_capture", common.open_video_capture)
        monkeypatch.setattr(quality_module, "open_video_capture", common.open_video_capture)

        entropy_df = pd.DataFrame(index=range(4))
        quality_df = pd.DataFrame(index=range(4))
        entropy_module.VisualEntropyFeature(config={}).run("video.mp4", {"data": entropy_df})
        quality_module.FrameQualityFeature(config={}).run("video.mp4", {"data": quality_df})

        assert entropy_df["visual_entropy"].notna().sum() == 3
        assert quality_df[["brightness", "sharpness", "visual_complexity"]].notna().sum().min() == 3


class TestAudioFeatures:
    def test_speech_music_silence_and_background_music_features(self, monkeypatch):
        _install_audio_stubs(monkeypatch)
        module = load_module("src.extractors.audio.speech_music_silence_feature", "src/extractors/audio/speech_music_silence_feature.py")

        out = module.extract_speech_music_silence(np.array([0.5, 0.0, 0.0, 0.0, 0.4]), np.array([0.1, 0.0, 0.0, 0.0, 0.8]), window_sec=3, silence_rms_threshold=0.01)
        bgm = module.extract_background_music_features(np.array([0.0, 0.2, 0.5]), np.array([10.0, 20.0, 80.0]), np.array([5.0, 15.0, 60.0]), change_percentile=50.0)

        assert set(out.columns) == {"speech_ratio", "silence_stretch", "music_only"}
        assert out["silence_stretch"].iloc[1:4].tolist() == [1.0, 1.0, 1.0]
        assert bgm["has_background_music"].tolist() == [0.0, 1.0, 1.0]
        assert bgm["music_changed"].iloc[-1] == 1.0

    def test_loudness_dynamics_and_skip_contract(self, monkeypatch):
        _install_audio_stubs(monkeypatch)
        module = load_module("src.extractors.audio.loudness_dynamics_feature", "src/extractors/audio/loudness_dynamics_feature.py")
        monkeypatch.setattr(module, "get_video_duration", lambda _path: 4)
        monkeypatch.setattr(module, "load_video_audio", lambda *_a, **_k: (np.ones(4096, dtype=np.float32), 22050))
        monkeypatch.setattr(module.librosa.feature, "rms", lambda **_k: np.array([[0.1, 0.2, 0.4, 0.2, 0.1]], dtype=np.float64))

        skipped = module.extract_loudness_dynamics("video.mp4", config={}, existing_features=["loudness_change", "loudness_variance"])
        df = module.extract_loudness_dynamics("video.mp4", config={})

        assert skipped.empty
        assert list(df.columns) == ["loudness_change", "loudness_variance"]
        assert len(df) == 4
        assert df["loudness_change"].max() >= 0

    def test_loudness_novelty_handles_missing_audio_and_detects_spikes(self, monkeypatch):
        _install_audio_stubs(monkeypatch)
        module = load_module("src.extractors.audio.loudness_novelty_feature", "src/extractors/audio/loudness_novelty_feature.py")
        monkeypatch.setattr(module, "video_duration_seconds", lambda _path: 6)
        monkeypatch.setattr(module, "load_video_audio_or_none", lambda *_a, **_k: None)

        missing = module.extract_loudness_novelty("video.mp4", config={})

        assert set(missing.columns) == {"loudness_zscore", "loudness_spike", "loudness_drop"}
        assert float(missing.to_numpy().sum()) == 0.0

        monkeypatch.setattr(module, "load_video_audio_or_none", lambda *_a, **_k: (np.ones(4096, dtype=np.float32), 22050))
        monkeypatch.setattr(module.librosa.feature, "rms", lambda **_k: np.array([[0.1, 0.1, 0.1, 2.0, 0.1, 0.1]], dtype=np.float64))
        detected = module.extract_loudness_novelty("video.mp4", config={})

        assert len(detected) == 6
        assert detected["loudness_zscore"].abs().max() > 0
