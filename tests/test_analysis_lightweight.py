import sys
import types

import numpy as np
import pandas as pd
import pytest

from tests.helpers import load_module


fi_utils = load_module("analysis.feature_importance.utils", "analysis/feature_importance/utils.py")
corr = load_module("analysis.feature_importance.correlation_analysis", "analysis/feature_importance/correlation_analysis.py")
norm_curves = load_module("src.normalize.curves", "src/normalize/curves.py")


class TestFeatureImportanceUtils:
    def test_load_features_csv_adds_time_seconds(self, tmp_path):
        path = tmp_path / "v1_features.csv"
        path.write_text("time,retention,wps\n00:00:00,100,1\n00:00:02,80,3\n", encoding="utf-8")

        df = fi_utils.load_features_csv(path)

        assert "time_sec" in df.columns
        np.testing.assert_allclose(df["time_sec"].values, [0.0, 2.0])

    def test_load_all_videos_uses_feature_file_names(self, tmp_path):
        (tmp_path / "a_features.csv").write_text("retention,wps\n100,1\n", encoding="utf-8")
        (tmp_path / "b_features.csv").write_text("retention,wps\n90,2\n", encoding="utf-8")

        videos = fi_utils.load_all_videos(tmp_path)

        assert set(videos) == {"a", "b"}
        assert videos["a"]["wps"].iloc[0] == 1

    def test_load_all_videos_errors_without_feature_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            fi_utils.load_all_videos(tmp_path)

    def test_aggregate_per_video_adds_targets(self):
        videos = {
            "v1": pd.DataFrame({"time": ["0", "1"], "retention": [100.0, 80.0], "wps": [1.0, 3.0], "label": ["x", "y"]}),
            "v2": pd.DataFrame({"time": ["0", "1"], "retention": [50.0, 25.0], "wps": [2.0, 4.0], "label": ["x", "y"]}),
        }

        out = fi_utils.aggregate_per_video(videos)

        assert list(out.index) == ["v1", "v2"]
        assert out.loc["v1", "wps"] == 2.0
        assert out.loc["v1", "target_avg_retention"] == 90.0
        assert out.loc["v1", "target_drop_rate"] == 0.2
        assert "label" not in out.columns

    def test_aggregate_per_video_supports_median_std_max_and_rejects_unknown(self):
        videos = {"v": pd.DataFrame({"retention": [100.0, 50.0], "x": [1.0, 3.0]})}

        assert fi_utils.aggregate_per_video(videos, agg="median").loc["v", "x"] == 2.0
        assert fi_utils.aggregate_per_video(videos, agg="max").loc["v", "x"] == 3.0
        assert fi_utils.aggregate_per_video(videos, agg="std").loc["v", "x"] > 0
        with pytest.raises(ValueError):
            fi_utils.aggregate_per_video(videos, agg="bad")

    def test_prepare_x_y_and_feature_group(self):
        df = pd.DataFrame({"wps": [1.0, 2.0], "speaker_prob": [0.1, 0.2], "target_avg_retention": [80.0, np.nan], "note": ["a", "b"]})

        x, y = fi_utils.prepare_X_y(df)

        assert list(x.columns) == ["wps", "speaker_prob"]
        assert list(y.values) == [80.0]
        assert fi_utils.feature_group_of("wps") == "audio_speech"
        assert fi_utils.feature_group_of("unknown_feature") == "unknown"

    def test_save_importance_csv_sorts_and_creates_parent(self, tmp_path):
        out = tmp_path / "nested" / "importance.csv"
        df = pd.DataFrame({"feature": ["a", "b"], "importance": [0.1, 0.9]})

        fi_utils.save_importance_csv(df, out, sort_by="importance")

        saved = pd.read_csv(out)
        assert saved["feature"].tolist() == ["b", "a"]


class TestCorrelationAnalysis:
    def test_compute_feature_target_correlation_orders_by_abs_value(self):
        x = pd.DataFrame({"positive": [1, 2, 3, 4], "negative": [4, 3, 2, 1], "flat": [1, 1, 1, 1]})
        y = pd.Series([1, 2, 3, 4])

        out = corr.compute_feature_target_correlation(x, y, method="spearman")

        assert out.iloc[0]["feature"] in {"positive", "negative"}
        assert set(out["method"]) == {"spearman"}
        assert out.loc[out["feature"] == "flat", "abs_correlation"].iloc[0] == 0.0

    def test_compute_per_second_correlation_pools_videos(self):
        videos = {
            "a": pd.DataFrame({"retention": [1, 2, 3], "wps": [1, 2, 3], "time": ["0", "1", "2"]}),
            "b": pd.DataFrame({"retention": [3, 2, 1], "wps": [3, 2, 1], "time": ["0", "1", "2"]}),
        }

        out = corr.compute_per_second_correlation(videos)

        assert out.loc[out["feature"] == "wps", "correlation"].iloc[0] > 0.9

    def test_per_second_correlation_errors_without_target(self):
        with pytest.raises(ValueError):
            corr.compute_per_second_correlation({"a": pd.DataFrame({"x": [1, 2]})})

    def test_correlation_matrix_and_redundancy(self):
        x = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6], "c": [3, 2, 1]})

        matrix = corr.compute_feature_correlation_matrix(x)
        pairs = corr.find_redundant_features(matrix, threshold=0.99)

        assert matrix.loc["a", "a"] == 1.0
        assert ("a", "b", 1.0) in pairs

    def test_mutual_information_returns_feature_rows(self):
        x = pd.DataFrame({"x": [0, 1, 2, 3], "z": [1, 1, 1, 1]})
        y = pd.Series([0, 1, 2, 3])

        out = corr.compute_mutual_information(x, y, random_state=0)

        assert set(out["feature"]) == {"x", "z"}
        assert set(out["method"]) == {"mutual_information"}


class TestNormalizeCurves:
    def test_clip_unit_interval_and_soft_non_increasing(self):
        clipped = norm_curves.clip_unit_interval(np.array([-1.0, 0.2, 2.0]))
        np.testing.assert_allclose(clipped, [0.0, 0.2, 1.0])

        softened = norm_curves.soft_non_increasing(np.array([1.0, 1.5, 1.7, 0.5]), max_increase=0.1)
        np.testing.assert_allclose(softened, [1.0, 1.0, 1.0, 0.5])

    def test_smooth_max_step_limits_large_jumps(self):
        out = norm_curves.smooth_max_step(np.array([0.0, 10.0, -10.0]), max_step=2.0)

        np.testing.assert_allclose(out, [0.0, 2.0, 0.0])

    def test_savgol_pipeline_preserves_length(self):
        curve = np.linspace(0.0, 1.0, 9)

        out = norm_curves.smooth_curve_savgol_then_max_step(curve, max_step=0.5, savgol_window=5, savgol_order=2)

        assert len(out) == len(curve)

class TestRetentionAnalysisCurrentApi:
    def _load_module(self, monkeypatch):
        cv2 = types.ModuleType("cv2")
        monkeypatch.setitem(sys.modules, "cv2", cv2)
        return __import__("src.retention_analysis", fromlist=["*"])

    def test_basic_retention_metrics(self, monkeypatch):
        module = self._load_module(monkeypatch)
        retention = np.array([100.0, 50.0, 0.0])

        assert module.compute_avd_from_retention(retention, duration_seconds=10) == pytest.approx(5.0)
        assert module.compute_retention_at(retention, 1) == 50.0
        assert module.compute_retention_at(retention, 30) is None
        assert module.video_retention_metrics(retention, 10)["mean_retention"] == 50.0

    def test_resample_extrapolate_and_format_time(self, monkeypatch):
        module = self._load_module(monkeypatch)

        np.testing.assert_allclose(module._resample_retention(np.array([0.0, 10.0]), 3), [0.0, 5.0, 10.0])
        assert len(module._extrapolate_trend(np.array([10.0, 8.0]), 4)) == 4
        assert module.format_time(90) == "1:30"

    def test_compute_channel_baseline_strategies(self, monkeypatch):
        module = self._load_module(monkeypatch)
        data = [
            {"name": "a", "retention_series": np.array([100.0, 80.0, 60.0]), "duration_sec": 2.0},
            {"name": "b", "retention_series": np.array([100.0, 50.0]), "duration_sec": 1.0},
        ]

        mean_result = module.compute_channel_baseline(data, strategy="mean_duration")
        normalized = module.compute_channel_baseline(data, strategy="normalized_100")

        assert mean_result["n_videos"] == 2
        assert mean_result["target_length"] == 2
        assert normalized["target_length"] == module.NORMALIZED_BASELINE_POINTS
        with pytest.raises(ValueError):
            module.compute_channel_baseline([], strategy="mean_duration")
