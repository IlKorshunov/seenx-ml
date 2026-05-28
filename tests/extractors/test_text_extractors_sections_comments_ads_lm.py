import json
import re
import sys
import types

import numpy as np
import torch

from tests.extractors.test_text_extractors_more import _install_base_stubs, _segments
from tests.helpers import load_module


def _install_common_stub(monkeypatch, duration: int = 12):
    common = types.ModuleType("src.extractors.text.common")

    def collect_valid_segments_with_mid(segments, dur):
        out = []
        for segment in segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            start = max(0, int(segment["start"]))
            end = min(dur, int(np.ceil(segment["end"])))
            if start < end:
                out.append((text, start, end, ((start + end) / 2.0) / max(dur, 1)))
        return out

    common.collect_valid_segments_with_mid = collect_valid_segments_with_mid  # type: ignore[attr-defined]
    common.collect_valid_segment_dicts = lambda segments: [s for s in segments if (s.get("text") or "").strip()]  # type: ignore[attr-defined]
    common.release_models = lambda *a, **k: None  # type: ignore[attr-defined]
    common.video_id = lambda _path: "vid"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.extractors.text.common", common)
    return common


def _install_transformer_stubs(monkeypatch):
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **k: object())  # type: ignore[attr-defined]
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: object())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_section_boundaries_and_extract_with_mocked_models(monkeypatch):
    segments = _segments(
        ("Всем привет, в этом видео поговорим про ML", 0.0, 2.0),
        ("Начнем с того, как устроены данные", 2.0, 4.0),
        ("Основная часть про признаки и обучение", 4.0, 7.0),
        ("Спасибо за просмотр, подписывайтесь", 9.0, 12.0),
    )
    _install_base_stubs(monkeypatch, segments, duration=12)
    _install_common_stub(monkeypatch)
    zeroshot = types.ModuleType("src.extractors.text._zeroshot")
    zeroshot.ZeroShotTask = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    zeroshot.classify_segments = lambda texts, *_a, **_k: np.linspace(0.2, 0.6, len(texts))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.extractors.text._zeroshot", zeroshot)
    _install_transformer_stubs(monkeypatch)
    module = load_module("src.extractors.text.section_feature", "src/extractors/text/section_feature.py")

    valid = module.collect_valid_segments_with_mid(segments, 12)
    assert "[2]" in module._format_segments(valid, start_idx=1)
    assert module._parse_llm_response("boundary is 2", 4) == 2
    assert module._parse_llm_response("segment 99", 4) is None

    monkeypatch.setattr(module, "_llm_generate", lambda *_a, **_k: "2")
    assert module._run_boundary_prompt(object(), object(), valid, module._INTRO_PROMPT, edge="start") == 4
    assert module._run_boundary_prompt(object(), object(), valid, module._OUTRO_PROMPT, edge="end") == 2

    intro_boundary = module._detect_boundary_nli(valid, 12, edge="start", config={})
    outro_boundary = module._detect_boundary_nli(valid, 12, edge="end", config={})
    assert intro_boundary >= 0
    assert 0 <= outro_boundary <= 12

    boosted = module._apply_regex_boost(module._build_intro_curve(12, 4), valid, module.RU_INTRO, edge="start")
    assert boosted[0] >= 0.85
    assert module._apply_position_boost(np.zeros(12), 12, edge="end")[-1] > 0.0

    monkeypatch.setattr(module, "_load_llm", lambda: (object(), object()))
    monkeypatch.setattr(module, "_unload_llm", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_run_boundary_prompt", lambda *_a, edge, **_k: 4 if edge == "start" else 9)
    monkeypatch.setattr(module, "_detect_boundary_nli", lambda *_a, edge, **_k: 3 if edge == "start" else 10)
    df = module.extract_sections("fake.mp4", config={})

    assert set(df.columns) == {"is_intro", "is_outro"}
    assert len(df) == 12
    assert df["is_intro"].iloc[0] > df["is_intro"].iloc[6]
    assert df["is_outro"].iloc[-1] > df["is_outro"].iloc[4]


def test_comment_features_from_json_and_error_branches(monkeypatch, tmp_path):
    _install_base_stubs(monkeypatch, _segments(("text", 0.0, 8.0)), duration=8)
    _install_common_stub(monkeypatch)
    module = load_module("src.extractors.text.comment_feature", "src/extractors/text/comment_feature.py")
    comments_root = tmp_path / "comments"
    comments_path = comments_root / "channel" / "vid" / "comments.json"
    comments_path.parent.mkdir(parents=True)
    comments_path.write_text(
        json.dumps(
            {
                "video_description": "00:01 Intro\n0:04 Topic",
                "threads": [
                    {
                        "text": "Круто, а почему в 0:02 так? великолепно",
                        "timecodes": [{"seconds": 2}],
                        "like_count": 8,
                        "replies": [{"author": "@ivanlyrics", "text": "спасибо"}],
                    },
                    {
                        "text": "Скучно и нудно 0:06",
                        "timecodes": [{"seconds": 6}],
                        "like_count": 1,
                        "replies": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_COMMENTS_ROOT", comments_root)

    assert module._parse_description_timecodes("0:01 one 01:02 two 1:02:03 three") == [1, 62, 3723]
    flattened = module._flatten_comments(json.loads(comments_path.read_text())["threads"])
    assert len(flattened) == 3
    assert module._find_comments_json("vid") == comments_path

    df = module.extract_comment_features("vid.mp4", config={})

    assert set(module._COLS).issubset(df.columns)
    assert df["desc_chapter_start"].iloc[1] == 1.0
    assert df["author_reply_rate_video"].iloc[0] == 0.5
    assert df["comment_positive_rate_30s"].max() > 0.0
    assert df["comment_aggression_rate_30s"].max() > 0.0
    assert df["complex_words_ratio_video"].iloc[0] > 0.0

    comments_path.write_text("{broken", encoding="utf-8")
    broken = module.extract_comment_features("vid.mp4", config={})
    assert broken.shape == (8, len(module._COLS))
    assert float(broken.to_numpy().sum()) == 0.0

    monkeypatch.setattr(module, "_COMMENTS_ROOT", tmp_path / "missing")
    missing = module.extract_comment_features("vid.mp4", config={})
    assert float(missing.to_numpy().sum()) == 0.0


def test_ad_segment_helpers_and_extract(monkeypatch):
    constants = {
        "RU_AD_PATTERNS": re.compile(r"промокод|скидк", re.IGNORECASE),
        "RU_AD_CTA_PATTERNS": re.compile(r"ссылка в описании", re.IGNORECASE),
    }
    segments = _segments(
        ("основная тема", 0.0, 2.0),
        ("у партнера есть промокод", 2.0, 4.0),
        ("ссылка в описании", 4.0, 6.0),
        ("снова контент", 6.0, 8.0),
        ("скидка для зрителей", 8.0, 10.0),
    )
    _install_base_stubs(monkeypatch, segments, duration=10, constants=constants)
    _install_common_stub(monkeypatch)
    _install_transformer_stubs(monkeypatch)
    module = load_module("src.extractors.text.ad_segment_feature", "src/extractors/text/ad_segment_feature.py")

    valid = module.collect_valid_segment_dicts(segments)
    assert "[1]" in module._format_segments(valid)
    assert module._detect_ads_regex(valid) == [(1, 4)]
    assert module._merge_ranges([(0, 1), (4, 5), (10, 10)]) == [(0, 5), (10, 10)]

    monkeypatch.setattr(module, "_load_llm", lambda: (object(), object()))
    monkeypatch.setattr(module, "_unload_llm", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_llm_generate", lambda *_a, **_k: "2-3, 5-5")
    assert module._detect_ads_llm(valid) == [(1, 4)]

    monkeypatch.setattr(module, "_detect_ads_llm", lambda _segments: [(1, 2)])
    df = module.extract_ad_segments("fake.mp4", config={})

    assert list(df.columns) == ["is_ad", "ad_segment_length"]
    assert df["is_ad"].iloc[2:10].eq(1.0).all()
    assert df["ad_segment_length"].max() == 8.0


class _FakeLM(torch.nn.Module):
    def forward(self, input_ids):
        vocab_size = 12
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab_size, dtype=torch.float32)
        logits[..., 1] = 2.0
        logits[..., 2] = 0.5
        return types.SimpleNamespace(logits=logits)


class _FakeTokenizer:
    is_fast = False
    model_max_length = 4

    def __call__(self, *_args, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)}


def test_speech_lm_surprisal_helpers_and_extract(monkeypatch):
    segments = [
        {
            "text": "fallback text",
            "start": 0.0,
            "end": 3.0,
            "words": [
                {"word": "hello", "start": 0.2},
                {"text": "world", "start": 1.4},
                {"word": "", "start": 2.0},
            ],
        }
    ]
    _install_base_stubs(monkeypatch, segments, duration=4)
    module = load_module("src.extractors.text.speech_lm_surprisal_feature", "src/extractors/text/speech_lm_surprisal_feature.py")

    text, char_sec = module._build_text_and_char_sec(segments, 4)
    assert text == "hello world"
    assert set(char_sec.tolist()) == {0, 1}
    fallback_text, fallback_sec = module._build_text_and_char_sec([{"text": "plain", "start": 2.1}], 4)
    assert fallback_text == "plain"
    assert fallback_sec.tolist() == [2, 2, 2, 2, 2]
    assert module._build_text_and_char_sec([{"text": "   "}], 2)[0] == ""

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    one_pass = module._one_pass_surprisal(input_ids[:, :3], _FakeLM())
    sliding = module._sliding_surprisal(input_ids, _FakeLM(), max_len=4, stride=2)
    assert one_pass.shape == (3,)
    assert sliding.shape == (6,)
    assert float(sliding[1:].mean()) > 0.0

    monkeypatch.setattr(module, "_get_lm", lambda _model_id, _device: (_FakeLM(), _FakeTokenizer()))
    df = module.extract_speech_lm_surprisal("fake.mp4", config={"device": "cpu", "lm_surprisal_model_id": "fake"})

    assert list(df.columns) == ["speech_lm_surprisal", "speech_lm_surprisal_vel"]
    assert len(df) == 4
    assert np.isfinite(df.to_numpy()).all()
    assert df["speech_lm_surprisal"].mean() > 0.0
    assert module.extract_speech_lm_surprisal("fake.mp4", config={}, existing_features=list(module._COLS)).empty
