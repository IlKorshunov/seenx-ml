import sys
import types

import numpy as np
import pandas as pd
import torch

from tests.extractors.test_text_extractors_more import _install_base_stubs, _segments
from tests.helpers import load_module


class _Encoded(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, *_a, **_k):
        return cls()

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return _Encoded({"attention_mask": torch.ones((len(texts), 4), dtype=torch.float32)})


class _FakeModel(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, *_a, **_k):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return self

    def forward(self, **encoded):
        n = encoded["attention_mask"].shape[0]
        base = torch.arange(n * 4 * 8, dtype=torch.float32).reshape(n, 4, 8)
        return types.SimpleNamespace(last_hidden_state=base + 1.0)


def _install_transformers(monkeypatch):
    transformers = types.ModuleType("transformers")
    transformers.AutoModel = _FakeModel  # type: ignore[attr-defined]
    transformers.AutoTokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def _install_common(monkeypatch):
    common = types.ModuleType("src.extractors.text.common")
    common.EMBEDDINGS_ROOT = "embeddings"
    common.video_id = lambda _path: "vid"  # type: ignore[attr-defined]
    common.release_models = lambda *a, **k: None  # type: ignore[attr-defined]
    common.load_segment_embeddings = lambda *_a, **_k: (None, [])  # type: ignore[attr-defined]

    def collect_valid_segment_dicts(segments):
        return [segment for segment in segments if (segment.get("text") or "").strip()]

    def collect_valid_segments(segments, duration):
        out = []
        for segment in segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            start = max(0, int(np.floor(segment["start"])))
            end = min(duration, int(np.ceil(segment["end"])))
            if start < end:
                out.append((text, start, end))
        return out

    common.collect_valid_segment_dicts = collect_valid_segment_dicts  # type: ignore[attr-defined]
    common.collect_valid_segments = collect_valid_segments  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.extractors.text.common", common)
    return common


def test_chapter_feature_with_saved_embeddings_and_fallback(monkeypatch):
    segments = _segments(
        ("intro", 0.0, 2.0),
        ("same topic", 2.0, 4.0),
        ("same topic more", 4.0, 6.0),
        ("new subject", 6.0, 8.0),
        ("new subject more", 8.0, 10.0),
        ("new subject finish", 10.0, 12.0),
    )
    _install_base_stubs(monkeypatch, segments, duration=12)
    common = _install_common(monkeypatch)
    _install_transformers(monkeypatch)
    module = load_module("src.extractors.text.chapter_feature", "src/extractors/text/chapter_feature.py")

    embs = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.95, 0.05],
            [-0.9, 0.1],
        ],
        dtype=np.float32,
    )
    common.load_segment_embeddings = lambda *_a, **_k: (embs, [{"start": 0}])  # type: ignore[attr-defined]
    monkeypatch.setattr(module, "_load_saved_embeddings", lambda _video_path: (embs, [{"start": 0}]))
    monkeypatch.setattr(module, "MIN_CHAPTER_SEGMENTS", 2)

    df = module.extract_chapters("vid.mp4", config={"device": "cpu"})

    assert set(df.columns) == {"chapter_id", "n_chapters", "topic_change_rate"}
    assert df["n_chapters"].iloc[0] >= 2.0
    assert df["chapter_id"].iloc[-1] >= 1.0
    assert df["topic_change_rate"].max() > 0.0

    _install_base_stubs(monkeypatch, _segments(("one", 0.0, 2.0)), duration=2)
    _install_common(monkeypatch)
    module = load_module("src.extractors.text.chapter_feature_short", "src/extractors/text/chapter_feature.py")
    short = module.extract_chapters("vid.mp4", config={"device": "cpu"})
    assert short["n_chapters"].eq(1.0).all()


def test_clickbait_gap_rules_and_extraction(monkeypatch, tmp_path):
    segments = _segments(("объясняю обычный контент", 0.0, 10.0), ("дальше детали", 10.0, 20.0))
    _install_base_stubs(monkeypatch, segments, duration=20)
    _install_common(monkeypatch)
    _install_transformers(monkeypatch)
    module = load_module("src.extractors.text.clickbait_gap_feature", "src/extractors/text/clickbait_gap_feature.py")
    comments_path = tmp_path / "comments" / "channel" / "vid" / "comments.json"
    comments_path.parent.mkdir(parents=True)
    comments_path.write_text('{"video_title": "ШОК секрет 100 фактов", "video_description": "лучшая правда"}', encoding="utf-8")
    monkeypatch.setattr(module, "_COMMENTS_ROOT", tmp_path / "comments")

    assert module._find_comments_json("vid") == comments_path
    assert module._title_claim_intensity("ШОК секрет 100 фактов", "лучшая правда") > 0.5

    df = module.extract_clickbait_gap("vid.mp4", config={"device": "cpu"})
    assert set(df.columns) == module._COLS
    assert len(df) == 20
    assert df["title_claim_intensity"].iloc[0] > 0.5
    assert np.isfinite(df["title_transcript_gap"]).all()

    monkeypatch.setattr(module, "_load_title_desc", lambda _vid: ("", ""))
    zeros = module.extract_clickbait_gap("vid.mp4", config={"device": "cpu"})
    assert zeros["title_transcript_gap"].eq(0.0).all()


def test_friction_parsing_windows_and_empty_branch(monkeypatch):
    segments = _segments(("жаргон и абстракция", 0.0, 2.0), ("повтор темы", 2.0, 4.0), ("норма", 4.0, 6.0))
    _install_base_stubs(monkeypatch, segments, duration=6)
    _install_common(monkeypatch)
    _install_transformers(monkeypatch)
    module = load_module("src.extractors.text.friction_feature", "src/extractors/text/friction_feature.py")

    parsed = module._parse_scores("[1] 1.5,0.2,0.4,0.6\nbad\n[3] 0.1,0.2,0.3,0.4")
    assert parsed[1] == (1.0, 0.2, 0.4, 0.6)
    assert "[1]" in module._format_segments(module.collect_valid_segment_dicts(segments), start_idx=1)

    monkeypatch.setattr(module, "_load_llm", lambda: (object(), object()))
    monkeypatch.setattr(module, "_llm_generate", lambda *_a, **_k: "[1] 0.8,0.2,0.4,0.0\n[2] 0.1,0.9,0.3,0.2")
    monkeypatch.setattr(module, "release_models", lambda *a, **k: None)
    df = module.extract_friction("vid.mp4", config={})

    assert set(df.columns) == module._COLS
    assert df["friction_jargon"].iloc[0] == 0.8
    assert df["friction_repetition"].iloc[2] == 0.9
    assert df["friction_total"].iloc[0] > 0.0

    _install_base_stubs(monkeypatch, [], duration=3)
    _install_common(monkeypatch)
    module = load_module("src.extractors.text.friction_feature_empty", "src/extractors/text/friction_feature.py")
    empty = module.extract_friction("vid.mp4", config={})
    assert empty.shape == (3, len(module._COLS))
    assert empty.to_numpy().sum() == 0.0


def test_hook_score_rules_with_mocked_question_embedding(monkeypatch):
    constants = {
        "ADDRESS_WINDOW_SEC": 25.0,
        "CLAIM_WINDOW_SEC": 25.0,
        "QUESTION_WINDOW_SEC": 25.0,
        "RU_ADDRESS": __import__("re").compile(r"привет|вы", __import__("re").IGNORECASE),
        "RU_CLAIMS": __import__("re").compile(r"секрет|узнаете", __import__("re").IGNORECASE),
        "NUMBER_PATTERN": __import__("re").compile(r"\d{2,}"),
        "HOOK_QUESTION_W": 0.25,
        "HOOK_ADDRESS_W": 0.15,
        "HOOK_CLAIM_W": 0.30,
        "HOOK_NUMBERS_W": 0.10,
        "HOOK_DENSITY_W": 0.20,
    }
    segments = _segments(("Привет, вы узнаете секрет 100?", 0.0, 3.0), ("обычный текст", 3.0, 6.0))
    _install_base_stubs(monkeypatch, segments, duration=6, constants=constants)
    _install_common(monkeypatch)
    _install_transformers(monkeypatch)
    module = load_module("src.extractors.text.hook_score_feature", "src/extractors/text/hook_score_feature.py")

    assert module._is_question_by_rules("Почему это важно")
    assert module._is_question_by_rules("Так работает?")
    assert not module._is_question_by_rules("обычный текст")

    monkeypatch.setattr(module, "_compute_is_question", lambda _segments, dur, _config: np.r_[np.ones(3), np.zeros(dur - 3)])
    df = module.extract_hook_score("vid.mp4", config={"device": "cpu"})

    assert set(df.columns) == {"hook_score", "hook_has_address", "is_question"}
    assert df["hook_score"].iloc[0] > 0.0
    assert df["hook_has_address"].iloc[0] == 1.0
    assert df["is_question"].iloc[0] == 1.0


def test_text_complexity_with_fake_spacy(monkeypatch):
    class Token:
        def __init__(self, text, children=None):
            self.text = text
            self.children = children or []
            self.is_alpha = text.isalpha()

    class Sent:
        def __init__(self, root):
            self.root = root

    class Doc:
        def __init__(self, words):
            child = Token(words[-1]) if words else Token("x")
            self._tokens = [Token(word) for word in words]
            self.sents = [Sent(Token(words[0] if words else "x", [child]))]

        def __iter__(self):
            return iter(self._tokens)

    class NLP:
        def pipe(self, texts, batch_size=64):
            return [Doc(text.split()) for text in texts]

    spacy = types.ModuleType("spacy")
    spacy.load = lambda *_a, **_k: NLP()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", spacy)
    constants = {
        "TEXT_COMPLEXITY_COLS": {"syntactic_depth", "lexical_diversity", "avg_word_length", "speech_complexity"},
        "TEXT_COMPLEXITY_MATTR_WINDOW": 3,
        "TEXT_COMPLEXITY_MIN_WORDS_FOR_DEPTH": 2,
        "TEXT_COMPLEXITY_WINDOW_SEC": 4,
    }
    segments = _segments(("простые слова тест", 0.0, 2.0), ("оченьразнообразные длинныеслова тут", 3.0, 5.0))
    _install_base_stubs(monkeypatch, segments, duration=6, constants=constants)
    module = load_module("src.extractors.text.text_complexity_feature", "src/extractors/text/text_complexity_feature.py")

    assert module._tree_depth(Token("root", [Token("child", [Token("leaf")])])) == 2
    assert module._mattr(["a", "b", "a"], window=2) == 1.0
    np.testing.assert_allclose(module.normalize_masked(np.array([1.0, 2.0]), np.array([True, False])), [0.0, 0.0])

    df = module.extract_text_complexity("vid.mp4", config={})

    assert set(df.columns) == constants["TEXT_COMPLEXITY_COLS"]
    assert df["avg_word_length"].max() > 0.0
    assert np.isfinite(df.to_numpy()).all()
