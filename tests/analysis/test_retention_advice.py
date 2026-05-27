from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest

from analysis import retention_advice as advice


def test_load_ranked_features_handles_missing_bad_and_sorted_files(tmp_path):
    assert advice._load_ranked_features(None) == []
    assert advice._load_ranked_features(tmp_path / "missing.csv") == []

    no_feature = tmp_path / "no_feature.csv"
    pd.DataFrame({"name": ["rms"]}).to_csv(no_feature, index=False)
    assert advice._load_ranked_features(no_feature) == []

    ranking = tmp_path / "ranking.csv"
    pd.DataFrame({"feature": ["edit_pace", "rms", None], "avg_rank": [2, 1, 3]}).to_csv(ranking, index=False)

    assert advice._load_ranked_features(ranking) == ["rms", "edit_pace"]


def test_find_prediction_matches_video_and_falls_back_when_video_column_absent(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    pd.DataFrame({"video": ["other"], "pred_retention": [1.0]}).to_csv(run_a / "holdout_prediction_vs_true.csv", index=False)
    pd.DataFrame({"video": ["target", "target"], "pred_retention": [3.0, 2.0]}).to_csv(run_b / "holdout_prediction_vs_true.csv", index=False)

    matched = advice._find_prediction(tmp_path, "target")

    assert matched["pred_retention"].tolist() == [3.0, 2.0]

    fallback_root = tmp_path / "fallback"
    fallback_root.mkdir()
    pd.DataFrame({"predicted": [0.8, 0.7]}).to_csv(fallback_root / "holdout_prediction_vs_true.csv", index=False)
    fallback = advice._find_prediction(fallback_root, "any")

    assert fallback["predicted"].tolist() == [0.8, 0.7]


def test_find_prediction_and_signal_report_clear_input_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Predictions root"):
        advice._find_prediction(tmp_path / "missing", "vid")

    pred_path = tmp_path / "holdout_prediction_vs_true.csv"
    pd.DataFrame({"video": ["other"], "pred_retention": [1.0]}).to_csv(pred_path, index=False)
    with pytest.raises(FileNotFoundError, match="No holdout prediction"):
        advice._find_prediction(tmp_path, "vid")

    with pytest.raises(ValueError, match="Prediction file must contain"):
        advice._signal(pd.DataFrame({"retention": [1.0]}))


def test_signal_interpolates_missing_prediction_values():
    out = advice._signal(pd.DataFrame({"pred": [1.0, np.nan, np.nan, 4.0]}))

    np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0])


def test_merge_segments_keeps_only_runs_with_minimum_length():
    mask = np.array([False, True, True, False, True, True, True])

    assert advice._merge_segments(mask, min_len=3) == [(4, 6)]


def test_find_segments_uses_effective_threshold_and_handles_short_curves():
    assert advice._find_segments(np.array([1.0, 0.9]), window=3, min_len=3, drop_threshold=-0.1, drop_percentile=20.0) == []

    y = np.array([100.0, 99.0, 98.0, 90.0, 82.0, 74.0, 73.0])
    segments = advice._find_segments(y, window=2, min_len=2, drop_threshold=-3.0, drop_percentile=100.0)

    assert segments[0][0:2] == (3, 6)
    assert segments[0][2] < -3.0
    assert segments[0][4] <= -3.0


def test_feature_cols_prefers_ranked_numeric_features_and_excludes_service_columns():
    df = pd.DataFrame(
        {
            "time": [0, 1],
            "retention": [100, 90],
            "rms": [0.1, 0.2],
            "edit_pace": [0.3, 0.4],
            "label": ["a", "b"],
            "viewer_address": [1, 0],
        }
    )

    cols = advice._feature_cols(df, ranked=["edit_pace", "missing", "rms"], top_n=1)

    assert cols[:2] == ["edit_pace", "rms"]
    assert "time" not in cols
    assert "retention" not in cols
    assert "label" not in cols


def test_advice_for_segment_returns_highest_severity_low_and_high_rules():
    df = pd.DataFrame(
        {
            "rms": [1.0, 0.9, 0.0, 0.0, 0.8, 0.9],
            "confusion": [0.0, 0.1, 1.0, 1.0, 0.2, 0.1],
            "unused_numeric": [100, 100, 100, 100, 100, 100],
        }
    )

    items = advice._advice_for_segment(df, start=2, end=3, ranked=["rms", "confusion"], top_n=2)

    assert [item["feature"] for item in items] == ["rms", "confusion"]
    assert items[0]["value"] == 0.0
    assert items[0]["median"] == pytest.approx(0.85)


def test_analyze_video_writes_empty_report_when_no_drop_segments(tmp_path, monkeypatch):
    features_dir = tmp_path / "features"
    preds_dir = tmp_path / "preds" / "run"
    out_dir = tmp_path / "out"
    features_dir.mkdir()
    preds_dir.mkdir(parents=True)

    pd.DataFrame({"rms": [0.1, 0.2, 0.3], "edit_pace": [0.5, 0.5, 0.5]}).to_csv(features_dir / "vid_features.csv", index=False)
    pd.DataFrame({"video": ["vid"] * 3, "pred_retention": [100.0, 100.0, 100.0]}).to_csv(preds_dir / "holdout_prediction_vs_true.csv", index=False)
    monkeypatch.setattr(advice, "_plot", lambda *args, **kwargs: None)

    segments = advice.analyze_video(
        features_dir / "vid_features.csv",
        ranked=[],
        args=Namespace(
            predictions_root=str(tmp_path / "preds"),
            output_dir=str(out_dir),
            window=2,
            min_len=2,
            drop_threshold=-1.0,
            drop_percentile=10.0,
            top_n=2,
        ),
    )

    assert segments == []
    report = json.loads((out_dir / "vid" / "advice.json").read_text(encoding="utf-8"))
    assert report["video_id"] == "vid"
    assert report["segments"] == []


def test_run_writes_summary_for_multiple_feature_files(tmp_path, monkeypatch):
    features_dir = tmp_path / "features"
    preds_dir = tmp_path / "preds" / "run"
    out_dir = tmp_path / "out"
    features_dir.mkdir()
    preds_dir.mkdir(parents=True)

    pd.DataFrame({"rms": [1.0, 0.0, 0.0, 0.0]}).to_csv(features_dir / "a_features.csv", index=False)
    pd.DataFrame({"rms": [1.0, 1.0, 0.0, 0.0]}).to_csv(features_dir / "b_features.csv", index=False)
    pd.DataFrame(
        {
            "video": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "pred_retention": [10.0, 9.0, 8.0, 7.0, 10.0, 10.0, 10.0, 10.0],
        }
    ).to_csv(preds_dir / "holdout_prediction_vs_true.csv", index=False)
    ranking = tmp_path / "ranking.csv"
    pd.DataFrame({"feature": ["rms"], "avg_rank": [1]}).to_csv(ranking, index=False)
    monkeypatch.setattr(advice, "_plot", lambda *args, **kwargs: None)

    advice.run(
        Namespace(
            features_dir=str(features_dir),
            predictions_root=str(tmp_path / "preds"),
            importance_path=str(ranking),
            output_dir=str(out_dir),
            top_n=1,
            window=1,
            min_len=2,
            drop_threshold=-0.5,
            drop_percentile=100.0,
        )
    )

    summary = pd.read_csv(out_dir / "summary.csv")

    assert list(summary.columns) == ["video_id", "start_sec", "end_sec", "retention_delta", "avg_derivative", "n_advice"]
