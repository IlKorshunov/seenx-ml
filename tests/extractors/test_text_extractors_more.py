"""Additional tests for lightweight text extractors."""

import re
import sys
import types

import numpy as np
import pandas as pd

from tests.helpers import load_module


def _install_base_stubs(monkeypatch, segments: list[dict] | None = None, duration: int = 5, constants: dict | None = None):
    for package_name in ("src", "src.extractors", "src.extractors.text", "src.utils"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    seenx = types.ModuleType("src.seenx_utils")
    seenx.get_video_duration = lambda _path: duration  # type: ignore[attr-defined]

    logger = types.ModuleType("src.utils.logger")
    logger.Logger = lambda show=True: types.SimpleNamespace(  # type: ignore[attr-defined]
        get_logger=lambda: types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    )

    config = types.ModuleType("src.utils.config")
    config.Config = dict  # type: ignore[attr-defined]

    transcript = types.ModuleType("src.utils.transcript_cache")
    transcript.get_transcript = lambda _path, _cfg: {"segments": segments or []}  # type: ignore[attr-defined]

    const_mod = types.ModuleType("src.extractors.text.constants")
    for key, value in (constants or {}).items():
        setattr(const_mod, key, value)

    for module_name, module in {
        "src.seenx_utils": seenx,
        "src.utils.logger": logger,
        "src.utils.config": config,
        "src.utils.transcript_cache": transcript,
        "src.extractors.text.constants": const_mod,
    }.items():
        monkeypatch.setitem(sys.modules, module_name, module)

    load_module("src.extractors.text._base", "src/extractors/text/_base.py")


def _segments(*items: tuple[str, float, float]) -> list[dict]:
    return [{"text": text, "start": start, "end": end} for text, start, end in items]


class TestViewerAddress:
    def test_marks_seconds_with_address_matches(self, monkeypatch):
        _install_base_stubs(
            monkeypatch,
            _segments(("ребята, подпишитесь на канал", 1.0, 3.0), ("обычный текст", 3.0, 5.0)),
            duration=6,
            constants={"VIEWER_ADDRESS_COLS": {"viewer_address"}, "VIEWER_ADDRESS_PATTERN": re.compile(r"\b(ребята|подпишитесь)\b", re.IGNORECASE)},
        )
        module = load_module("src.extractors.text.viewer_address_feature", "src/extractors/text/viewer_address_feature.py")

        df = module.extract_viewer_address("fake.mp4", config=None)

        assert list(df.columns) == ["viewer_address"]
        np.testing.assert_allclose(df["viewer_address"].values, [0, 2, 2, 0, 0, 0])

    def test_skips_when_feature_exists(self, monkeypatch):
        _install_base_stubs(
            monkeypatch,
            [],
            constants={"VIEWER_ADDRESS_COLS": {"viewer_address"}, "VIEWER_ADDRESS_PATTERN": re.compile("x")},
        )
        module = load_module("src.extractors.text.viewer_address_feature", "src/extractors/text/viewer_address_feature.py")

        assert module.extract_viewer_address("fake.mp4", config=None, existing_features=["viewer_address"]).empty


class TestSpeechFillers:
    def test_counts_fillers_per_segment(self, monkeypatch):
        _install_base_stubs(monkeypatch, _segments(("ну типа вот", 0.0, 2.0), ("без слов паразитов", 2.0, 4.0)), duration=4)
        module = load_module("src.extractors.text.speech_filler_feature", "src/extractors/text/speech_filler_feature.py")

        df = module.extract_speech_fillers("fake.mp4", config=None)

        assert list(df.columns) == ["crutch_cnt"]
        np.testing.assert_allclose(df["crutch_cnt"].values, [3, 3, 0, 0])

    def test_skip_contract(self, monkeypatch):
        _install_base_stubs(monkeypatch)
        module = load_module("src.extractors.text.speech_filler_feature", "src/extractors/text/speech_filler_feature.py")

        assert module.extract_speech_fillers("fake.mp4", config=None, existing_features=["crutch_cnt"]).empty


class TestSpeechIntelligibility:
    def test_averages_word_confidence_by_second(self, monkeypatch):
        segments = [
            {
                "text": "hello",
                "start": 0.0,
                "end": 2.0,
                "words": [
                    {"start": 0.1, "probability": 0.5},
                    {"start": 0.6, "probability": 1.0},
                    {"start": 1.2, "probability": 0.25},
                ],
            }
        ]
        _install_base_stubs(
            monkeypatch,
            segments,
            duration=3,
            constants={
                "SPEECH_INTELLIGIBILITY_COLS": {"speech_intelligibility", "speech_mumble_index"},
                "SPEECH_INTELLIGIBILITY_MUMBLE_SCALE": 1.0,
                "SPEECH_INTELLIGIBILITY_SMOOTH_WINDOW": 1,
                "SPEECH_INTELLIGIBILITY_WPS_FAST": 2.0,
            },
        )
        module = load_module("src.extractors.text.speech_intelligibility_feature", "src/extractors/text/speech_intelligibility_feature.py")

        df = module.extract_speech_intelligibility("fake.mp4", config=None)

        np.testing.assert_allclose(df["speech_intelligibility"].values, [0.75, 0.25, 1.0])
        assert df["speech_mumble_index"].iloc[0] > 0.0
        assert df["speech_mumble_index"].iloc[1] > df["speech_mumble_index"].iloc[0]

    def test_no_words_defaults_to_clear_speech(self, monkeypatch):
        _install_base_stubs(
            monkeypatch,
            [{"text": "silence", "start": 0.0, "end": 2.0, "words": []}],
            duration=2,
            constants={
                "SPEECH_INTELLIGIBILITY_COLS": {"speech_intelligibility", "speech_mumble_index"},
                "SPEECH_INTELLIGIBILITY_MUMBLE_SCALE": 1.0,
                "SPEECH_INTELLIGIBILITY_SMOOTH_WINDOW": 1,
                "SPEECH_INTELLIGIBILITY_WPS_FAST": 2.0,
            },
        )
        module = load_module("src.extractors.text.speech_intelligibility_feature", "src/extractors/text/speech_intelligibility_feature.py")

        df = module.extract_speech_intelligibility("fake.mp4", config=None)

        np.testing.assert_allclose(df["speech_intelligibility"].values, [1.0, 1.0])
        np.testing.assert_allclose(df["speech_mumble_index"].values, [0.0, 0.0])


class TestInformationDensity:
    def test_resamples_embeddings_by_segment_overlap(self, monkeypatch):
        _install_base_stubs(monkeypatch)
        common = types.ModuleType("src.extractors.text.common")
        common.EMBEDDINGS_ROOT = "embeddings"
        common.load_segment_embeddings = lambda *_a, **_k: (None, [])  # type: ignore[attr-defined]
        common.video_id = lambda path: "vid"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.extractors.text.common", common)
        module = load_module("src.extractors.text.information_density_feature", "src/extractors/text/information_density_feature.py")

        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        metadata = [{"start": 0.0, "end": 2.0}, {"start": 1.0, "end": 3.0}]

        out = module._resample_to_1fps(embeddings, metadata, duration=3)

        np.testing.assert_allclose(out[0], [1.0, 0.0])
        np.testing.assert_allclose(out[1], [0.5, 0.5])
        np.testing.assert_allclose(out[2], [0.0, 1.0])

    def test_missing_embeddings_returns_zero_features(self, monkeypatch):
        _install_base_stubs(monkeypatch, duration=3)
        common = types.ModuleType("src.extractors.text.common")
        common.EMBEDDINGS_ROOT = "embeddings"
        common.load_segment_embeddings = lambda *_a, **_k: (None, [])  # type: ignore[attr-defined]
        common.video_id = lambda path: "vid"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.extractors.text.common", common)
        module = load_module("src.extractors.text.information_density_feature", "src/extractors/text/information_density_feature.py")

        df = module.extract_information_density("fake.mp4", config=None)

        assert set(df.columns) == {"information_density", "cumulative_info"}
        assert (df.values == 0.0).all()


class TestTitleFeatures:
    def _load_title_module(self, monkeypatch):
        _install_base_stubs(monkeypatch, duration=4)
        common = types.ModuleType("src.extractors.text.common")
        common.release_models = lambda *a, **k: None  # type: ignore[attr-defined]
        common.video_id = lambda path: "vid"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.extractors.text.common", common)

        torch = types.ModuleType("torch")
        torch.bfloat16 = object()
        torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, *exc: None)  # type: ignore[attr-defined]
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForCausalLM = object  # type: ignore[attr-defined]
        transformers.AutoTokenizer = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "transformers", transformers)

        return load_module("src.extractors.text.title_feature", "src/extractors/text/title_feature.py")

    def test_regex_features_detect_number_question_and_caps(self, monkeypatch):
        module = self._load_title_module(monkeypatch)

        features = module._regex_features("КАК получить 10X РЕЗУЛЬТАТ?")

        assert features["title_has_number"] == 1.0
        assert features["title_has_question"] == 1.0
        assert features["title_caps_ratio"] > 0.0

    def test_extract_title_features_repeats_scores_for_duration(self, monkeypatch):
        module = self._load_title_module(monkeypatch)
        monkeypatch.setattr(module, "_load_title", lambda _video_id: "Как сделать 10 видео?")
        monkeypatch.setattr(
            module,
            "_llm_score",
            lambda _title: {
                "title_clickbait": 1.0,
                "title_clarity": 2.0,
                "title_emotional": 3.0,
                "title_specificity": 4.0,
                "title_urgency": 5.0,
            },
        )

        df = module.extract_title_features("fake.mp4", config=None)

        assert len(df) == 4
        assert (df["title_clickbait"] == 1.0).all()
        assert (df["title_has_number"] == 1.0).all()
