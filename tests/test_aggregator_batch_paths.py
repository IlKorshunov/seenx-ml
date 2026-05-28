from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def aggregator(monkeypatch):
    mod = pytest.importorskip("src.aggregator")
    monkeypatch.setattr(mod, "clear_transcript_cache", lambda: None)
    monkeypatch.setattr(mod, "_gpu_cleanup", lambda: None)
    return mod


def _acc() -> pd.DataFrame:
    idx = pd.to_timedelta(np.arange(5), unit="s")
    return pd.DataFrame({"retention": [100.0, 95.0, 90.0, 85.0, 80.0]}, index=idx)


def _video(tmp_path: Path) -> dict:
    return {"vid": "v", "video_path": str(tmp_path / "v" / "video.mp4"), "retention_path": str(tmp_path / "v" / "retention.csv"), "output_path": str(tmp_path / "v_features.csv")}


def _df(cols: list[str], n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({col: np.linspace(0.1, 0.9, n) for col in cols})


def test_checkpoint_failed_discovery_and_batch_load_helpers(tmp_path, monkeypatch, aggregator):
    vdir = tmp_path / "data" / "v"
    vdir.mkdir(parents=True)
    (vdir / "video.mp4").write_bytes(b"not-a-video")
    (vdir / "retention.csv").write_text("time_ratio,audience_watch_ratio\n0,1\n1,.5\n", encoding="utf-8")

    output_path = tmp_path / "out.csv"
    acc = _acc()
    aggregator._save_checkpoint(acc, str(output_path))
    loaded, features = aggregator._load_checkpoint(str(output_path))
    assert loaded.shape == acc.shape
    assert features == ["retention"]

    aggregator._save_failed_features(str(output_path), {"bad_b", "bad_a"})
    assert aggregator._load_failed_features(str(output_path)) == {"bad_a", "bad_b"}
    aggregator._save_failed_features(str(output_path), set())
    assert aggregator._load_failed_features(str(output_path)) == set()

    assert aggregator._discover_videos(str(tmp_path / "data"))[0]["vid"] == "v"
    assert aggregator._should_run({"a"}, only=None, existing=set())
    assert not aggregator._should_run({"a"}, only=None, existing={"a"})
    assert aggregator._should_run({"a"}, only={"a"}, existing={"a"})

    mapped = aggregator._batch_map_to_retention(acc.index, pd.DataFrame({"x": [0.0, 10.0]}))
    np.testing.assert_allclose(mapped["x"], [0.0, 2.5, 5.0, 7.5, 10.0])
    added = aggregator._batch_add_cols(acc.copy(), mapped, str(output_path))
    assert "x" in added.columns

    monkeypatch.setattr(aggregator, "get_retention", lambda video_path, retention_csv_path: acc)
    batch_acc, existing = aggregator._batch_load_video({"output_path": str(tmp_path / "fresh.csv"), "video_path": "v.mp4", "retention_path": "r.csv"})
    assert "retention" in existing
    assert batch_acc.index.name == "time"


def test_batch_finalize_derives_interactions_emotions_questions_and_dynamics(tmp_path, monkeypatch, aggregator):
    acc = _acc()
    acc["edit_pace"] = [1, 2, 3, 4, 5]
    acc["screencast_prob"] = [0, 1, 0, 1, 0]
    acc["is_ad"] = [0, 1, 0, 0, 1]
    acc["viewer_address"] = [1, 1, 0, 0, 1]
    acc["is_question"] = [0, 1, 0, 1, 0]
    acc["sent_joy"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    acc["sent_neutral"] = [0.5, 0.4, 0.3, 0.2, 0.1]
    acc["global_topic_dist"] = [1, 2, 3, 4, 5]
    acc["is_intro"] = [1, 1, 1, 1, 1]
    monkeypatch.setattr(aggregator, "fit_hill_curve", lambda time_sec, retention: (retention, np.array([0.0, 1.0, 1.0, 0.0])))

    out = aggregator._batch_finalize(acc, str(tmp_path / "features.csv"))

    assert "edit_pace_x_screencast" in out
    assert "is_ad_x_viewer_address" in out
    assert "question_density" in out
    assert {"ekman_joy", "ekman_neutral", "ekman_intensity"}.issubset(out.columns)
    assert "sent_joy" not in out.columns
    assert "global_topic_dist" not in out.columns
    assert out["is_intro"].iloc[-1] == 0
    assert "edit_pace_chg_5s" in out.columns
    assert (tmp_path / "features.csv").is_file()


def test_batch_run_group_for_video_covers_cpu_extractor_groups(tmp_path, monkeypatch, aggregator):
    v = _video(tmp_path)
    acc = _acc()
    config = {"device": "cpu"}

    monkeypatch.setattr(aggregator, "sound_features_pipeline", lambda **kwargs: _df(["rms", "zcr", "centroid", "rolloff"]))
    monkeypatch.setattr(aggregator, "extract_prosody", lambda **kwargs: _df(["pitch_mean", "pitch_std", "voiced_frac", "speech_rate_cv", "pause_rate"]))
    monkeypatch.setattr(aggregator, "extract_beat_sync", lambda **kwargs: _df(["beat_sync", "beat_sync_ratio"]))
    monkeypatch.setattr(aggregator, "extract_loudness_dynamics", lambda **kwargs: _df(["loudness_change", "loudness_variance"]))
    monkeypatch.setattr(aggregator, "extract_spectral_flux", lambda **kwargs: _df(["spectral_flux"]))
    monkeypatch.setattr(aggregator, "extract_laughter", lambda **kwargs: _df(["laughter_prob"]))
    monkeypatch.setattr(aggregator, "extract_sfx_energy", lambda **kwargs: _df(["sfx_energy"]))
    monkeypatch.setattr(aggregator, "extract_speech_predictability", lambda **kwargs: _df(["speech_predictability"]))
    monkeypatch.setattr(aggregator, "extract_speech_lm_surprisal", lambda **kwargs: _df(["speech_lm_surprisal", "speech_lm_surprisal_vel"]))
    monkeypatch.setattr(aggregator, "extract_object_density", lambda **kwargs: _df(["object_count", "unique_classes"]))
    monkeypatch.setattr(aggregator, "extract_video_intelligence", lambda **kwargs: _df(["content_rhythm", "visual_audio_sync", "narrative_momentum", "engagement_surprise"]))
    monkeypatch.setattr(aggregator, "extract_comment_features", lambda **kwargs: _df(sorted(aggregator._COMMENT_COLS)))
    monkeypatch.setattr(aggregator, "extract_speech_intelligibility", lambda **kwargs: _df(["speech_intelligibility", "speech_mumble_index"]))

    groups = ["sound", "audio_basic", "audio_extra", "object_density", "video_intelligence", "comment_social", "speech_intelligibility"]
    existing = set(acc.columns)
    for group in groups:
        acc, existing, failed = aggregator._batch_run_group_for_video(group, v, acc, existing, config)
        assert failed == set()

    expected = {"rms", "pitch_mean", "spectral_flux", "object_count", "content_rhythm", "speech_intelligibility"} | aggregator._COMMENT_COLS
    assert expected.issubset(acc.columns)


def test_batch_run_group_for_video_covers_text_and_demucs_groups(tmp_path, monkeypatch, aggregator):
    v = _video(tmp_path)
    acc = _acc()
    config = {"device": "cpu"}

    music = _df(["music_rms", "music_zcr", "music_centroid", "music_rolloff"])
    vocal = _df(["vocal_rms", "vocal_zcr", "vocal_centroid", "vocal_rolloff"])
    monkeypatch.setattr(aggregator, "get_vocal_music_features", lambda **kwargs: (music, vocal))
    monkeypatch.setattr(aggregator, "extract_wps", lambda **kwargs: _df(["wps"]))
    monkeypatch.setattr(aggregator, "extract_viewer_address", lambda **kwargs: _df(["viewer_address"]))
    monkeypatch.setattr(aggregator, "extract_speech_fillers", lambda **kwargs: _df(["crutch_cnt"]))
    monkeypatch.setattr(aggregator, "extract_ad_segments", lambda **kwargs: _df(["is_ad", "ad_segment_length"]))
    monkeypatch.setattr(aggregator, "extract_clickbait_gap", lambda **kwargs: _df(["title_transcript_gap", "title_delivery_30s", "title_claim_intensity"]))
    monkeypatch.setattr(aggregator, "extract_text_complexity", lambda **kwargs: _df(["syntactic_depth", "lexical_diversity", "avg_word_length", "speech_complexity"]))
    monkeypatch.setattr(aggregator, "extract_cultural_references", lambda **kwargs: _df(["has_person_mention", "has_org_mention"]))
    monkeypatch.setattr(aggregator, "extract_text_sentiment", lambda **kwargs: _df(sorted(aggregator._TEXT_EMOTION_COLS)))
    monkeypatch.setattr(aggregator, "extract_semantic_embeddings", lambda **kwargs: _df(["semantic_novelty", "topic_shift", "hook_similarity", "global_topic_dist", "semantic_momentum", "segment_self_similarity"]))
    monkeypatch.setattr(aggregator, "extract_hook_score", lambda **kwargs: _df(["hook_score", "hook_has_address", "is_question"]))
    monkeypatch.setattr(aggregator, "extract_chapters", lambda **kwargs: _df(["chapter_id", "n_chapters", "topic_change_rate"]))
    monkeypatch.setattr(aggregator, "extract_curiosity_gap", lambda **kwargs: _df(["curiosity_gap"]))
    monkeypatch.setattr(aggregator, "extract_topic_sharpness", lambda **kwargs: _df(["topic_sharpness_0_100"]))
    monkeypatch.setattr(aggregator, "extract_storytelling", lambda **kwargs: _df(["storytelling"]))
    monkeypatch.setattr(aggregator, "extract_viewer_engagement", lambda **kwargs: _df(["viewer_engagement"]))
    monkeypatch.setattr(aggregator, "extract_examples", lambda **kwargs: _df(["has_example"]))
    monkeypatch.setattr(aggregator, "extract_sections", lambda **kwargs: _df(["is_intro", "is_outro"]))
    monkeypatch.setattr(aggregator, "extract_information_density", lambda **kwargs: _df(["information_density", "cumulative_info"]))

    existing = set(acc.columns)
    for group in ["demucs", "whisper_text", "text_sentiment", "text_embedding", "text_zeroshot"]:
        acc, existing, failed = aggregator._batch_run_group_for_video(group, v, acc, existing, config)
        assert failed == set()

    expected = {"music_rms", "speech_ratio", "has_background_music", "wps", "sent_joy", "semantic_novelty", "curiosity_gap", "information_density"}
    assert expected.issubset(acc.columns)


def test_aggregate_batch_skips_groups_and_finalizes(tmp_path, monkeypatch, aggregator):
    data_dir = tmp_path / "data"
    vdir = data_dir / "v"
    vdir.mkdir(parents=True)
    (vdir / "video.mp4").write_bytes(b"")
    (vdir / "retention.csv").write_text("time_ratio,audience_watch_ratio\n0,1\n1,.8\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(aggregator, "get_retention", lambda video_path, retention_csv_path: _acc())
    monkeypatch.setattr(aggregator, "_BATCH_GROUPS", [("sound", {"rms"}, "0"), ("comment_social", aggregator._COMMENT_COLS, "0")])
    monkeypatch.setattr(
        aggregator,
        "_batch_run_group_for_video",
        lambda group_name, v, acc, existing, config, only=None: (pd.concat([acc, pd.DataFrame({"rms": np.arange(len(acc))}, index=acc.index)], axis=1), set(acc.columns) | {"rms"}, set()),
    )
    monkeypatch.setattr(aggregator, "_batch_finalize", lambda acc, output_path, skip_emotion_features=False: acc)

    results = aggregator.aggregate_batch(str(data_dir), str(out_dir), config={}, only={"rms"}, skip_comment_features=True)

    assert list(results) == ["v"]
    assert "rms" in results["v"].columns
