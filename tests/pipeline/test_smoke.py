import json
from argparse import Namespace

import pytest


def test_get_retention_builds_second_level_dataset(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    aggregator = pytest.importorskip("src.aggregator")

    retention_csv = tmp_path / "retention.csv"
    retention_csv.write_text("time_ratio,audience_watch_ratio\n0,1.0\n1,0.0\n", encoding="utf-8")
    monkeypatch.setattr(aggregator, "get_video_duration", lambda path: 4)

    df = aggregator.get_retention("video.mp4", str(retention_csv))

    assert list(df.columns) == ["retention"]
    assert df.index.name == "time"
    assert len(df) == 5
    assert df["retention"].tolist() == pytest.approx([100.0, 75.0, 50.0, 25.0, 0.0])


def test_aggregate_maps_selected_extractor_to_retention_grid(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    aggregator = pytest.importorskip("src.aggregator")

    retention_csv = tmp_path / "retention.csv"
    retention_csv.write_text("time_ratio,audience_watch_ratio\n0,1.0\n1,0.0\n", encoding="utf-8")
    output_path = tmp_path / "features.csv"
    monkeypatch.setattr(aggregator, "get_video_duration", lambda path: 4)
    monkeypatch.setattr(aggregator, "sound_features_pipeline", lambda **kwargs: pd.DataFrame({"rms": [0.0, 1.0]}))
    monkeypatch.setattr(aggregator, "clear_transcript_cache", lambda: None)

    df = aggregator.aggregate(
        video_path="video.mp4",
        audio_path="audio.mp3",
        output_path=str(output_path),
        config={},
        retention_csv_path=str(retention_csv),
        only={"rms"},
        skip_comment_features=True,
        skip_emotion_features=True,
    )

    assert list(df.columns) == ["retention", "rms"]
    assert df["retention"].tolist() == pytest.approx([100.0, 75.0, 50.0, 25.0, 0.0])
    assert df["rms"].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert not (tmp_path / "features.csv.partial").exists()


def test_retention_advice_writes_compact_report_and_ignores_ads(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    advice = pytest.importorskip("analysis.retention_advice")

    features_dir = tmp_path / "features"
    pred_dir = tmp_path / "experiments" / "run"
    out_dir = tmp_path / "advice"
    features_dir.mkdir()
    pred_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "retention": [100, 98, 96, 90, 85, 80, 75],
            "rms": [0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0],
            "is_ad": [0, 0, 1, 1, 1, 0, 0],
            "edit_pace": [0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(features_dir / "video_001_features.csv", index=False)
    pd.DataFrame({"video": ["video_001"] * 7, "pred_retention": [100, 99, 97, 92, 86, 80, 75]}).to_csv(pred_dir / "holdout_prediction_vs_true.csv", index=False)
    pd.DataFrame({"feature": ["is_ad", "rms", "edit_pace"], "avg_rank": [1, 2, 3]}).to_csv(tmp_path / "ranking.csv", index=False)
    monkeypatch.setattr(advice, "_plot", lambda *args, **kwargs: None)

    advice.run(
        Namespace(
            features_dir=str(features_dir),
            predictions_root=str(tmp_path / "experiments"),
            importance_path=str(tmp_path / "ranking.csv"),
            output_dir=str(out_dir),
            top_n=2,
            window=2,
            min_len=2,
            drop_threshold=-3.0,
            drop_percentile=100.0,
        )
    )

    report = json.loads((out_dir / "video_001" / "advice.json").read_text(encoding="utf-8"))
    segment = report["segments"][0]
    assert report["drop_criteria"]["threshold"] <= report["drop_criteria"]["fixed_threshold"]
    assert segment["start_sec"] == 3
    assert {item["feature"] for item in segment["advice"]} == {"rms", "edit_pace"}
