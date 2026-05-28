import types

import numpy as np
import pandas as pd
import pytest

from analysis.feature_importance import correlation_analysis as module


class _FakeSeaborn:
    @staticmethod
    def heatmap(*_args, **kwargs):
        ax = kwargs.get("ax")
        if ax is not None:
            ax.imshow(np.eye(2), vmin=0, vmax=1)
        return ax


def _sample_video_dfs():
    return {
        "a": pd.DataFrame({"time": ["00:00:00", "00:00:01"], "retention": [100.0, 80.0], "brightness": [0.1, 0.2], "wps": [2.0, 2.5]}),
        "b": pd.DataFrame({"time": ["00:00:00", "00:00:01"], "retention": [90.0, 60.0], "brightness": [0.3, 0.4], "wps": [1.0, 1.5]}),
        "c": pd.DataFrame({"time": ["00:00:00", "00:00:01"], "retention": [70.0, 50.0], "brightness": [0.5, 0.6], "wps": [3.0, 3.5]}),
        "d": pd.DataFrame({"time": ["00:00:00", "00:00:01"], "retention": [95.0, 75.0], "brightness": [0.7, 0.8], "wps": [2.7, 2.9]}),
    }


def test_correlation_helpers_cover_edge_cases():
    X = pd.DataFrame({"brightness": [1, 2, 3, 4], "wps": [4, 3, 2, 1], "constant": [1, 1, 1, 1], "nanish": [1, np.nan, 3, np.inf]})
    y = pd.Series([10, 20, 30, 40])

    spearman = module.compute_feature_target_correlation(X, y, method="spearman")
    pearson = module.compute_feature_target_correlation(X, y, method="pearson")
    assert spearman.iloc[0]["feature"] == "brightness"
    assert pearson.loc[pearson["feature"] == "constant", "correlation"].iloc[0] == 0.0
    assert pearson.loc[pearson["feature"] == "nanish", "p_value"].iloc[0] == 1.0

    one = module.compute_feature_correlation_matrix(pd.DataFrame({"only": [1, 2, 3]}))
    assert one.loc["only", "only"] == 1.0
    matrix = module.compute_feature_correlation_matrix(X[["brightness", "wps"]], method="pearson")
    assert matrix.loc["brightness", "wps"] < 0.0
    redundant = module.find_redundant_features(pd.DataFrame([[1.0, -0.9], [-0.9, 1.0]], columns=["a", "b"], index=["a", "b"]), threshold=0.85)
    assert redundant == [("a", "b", 0.9)]

    mi = module.compute_mutual_information(X[["brightness", "wps"]], y, n_neighbors=10)
    assert list(mi.columns) == ["feature", "mutual_information", "group", "method"]
    assert set(mi["feature"]) == {"brightness", "wps"}


def test_per_second_correlation_and_no_data_branch():
    result = module.compute_per_second_correlation(_sample_video_dfs(), target_col="retention")
    assert {"brightness", "wps"} <= set(result["feature"])

    with pytest.raises(ValueError, match="No valid data"):
        module.compute_per_second_correlation({"x": pd.DataFrame({"feature": [1, 2, 3]})})


def test_plotting_and_combined_ranking(tmp_path, monkeypatch):
    corr_df = pd.DataFrame({"feature": ["brightness", "wps"], "correlation": [0.8, -0.5], "abs_correlation": [0.8, 0.5]})
    out_bar = tmp_path / "bar.png"
    module.plot_correlation_bar(corr_df, "bar", out_bar, top_n=2)
    assert out_bar.exists()

    monkeypatch.setattr(module, "sns", _FakeSeaborn())
    matrix = pd.DataFrame([[1.0, 0.7], [0.7, 1.0]], columns=["brightness", "wps"], index=["brightness", "wps"])
    module.plot_correlation_heatmap(matrix, tmp_path / "heat.png")
    module.plot_group_heatmap(matrix, tmp_path / "group.png")
    module.plot_redundancy_network([], tmp_path / "none.png")
    module.plot_redundancy_network([("a", "b", 0.91)], tmp_path / "redundant.png", threshold=0.9)
    assert (tmp_path / "heat.png").exists()
    assert (tmp_path / "group.png").exists()
    assert (tmp_path / "redundant.txt").read_text().count("0.910") == 1

    mi = pd.DataFrame({"feature": ["wps", "brightness"], "mutual_information": [0.4, 0.2]})
    combined = module._build_combined_ranking(corr_df, corr_df, mi)
    assert combined.iloc[0]["feature"] == "brightness"
    assert {"rank_spearman", "rank_pearson", "rank_mi", "avg_rank"} <= set(combined.columns)


def test_run_correlation_analysis_with_synthetic_data(tmp_path, monkeypatch):
    video_dfs = _sample_video_dfs()
    monkeypatch.setattr(module, "load_all_videos", lambda _output_dir: video_dfs)
    monkeypatch.setattr(module, "default_output_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "plot_correlation_bar", lambda *a, **k: None)
    monkeypatch.setattr(module, "plot_correlation_heatmap", lambda *a, **k: None)
    monkeypatch.setattr(module, "plot_group_heatmap", lambda *a, **k: None)
    monkeypatch.setattr(module, "plot_redundancy_network", lambda *a, **k: None)
    monkeypatch.setattr(module, "compute_mutual_information", lambda X, y: pd.DataFrame({"feature": X.columns, "mutual_information": np.arange(len(X.columns)), "group": "x"}))

    results = module.run_correlation_analysis(output_dir="synthetic", results_dir=None, top_n=2, redundancy_threshold=0.2)

    assert {"spearman_agg", "pearson_agg", "spearman_ts", "mutual_info", "corr_matrix"} <= set(results)
    out_dir = tmp_path / "correlation"
    assert (out_dir / "spearman_agg.csv").exists()
    assert (out_dir / "combined_ranking.csv").exists()
    assert (out_dir / "redundant_pairs.csv").exists()


def test_run_correlation_analysis_handles_timeseries_failure(tmp_path, monkeypatch):
    video_dfs = _sample_video_dfs()
    monkeypatch.setattr(module, "load_all_videos", lambda _output_dir: video_dfs)
    monkeypatch.setattr(module, "aggregate_per_video", lambda _dfs: pd.DataFrame({"brightness": [1, 2, 3], "target_avg_retention": [1.0, 2.0, 3.0]}))
    monkeypatch.setattr(module, "default_output_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "compute_per_second_correlation", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(module, "compute_mutual_information", lambda X, y: pd.DataFrame({"feature": X.columns, "mutual_information": [1.0], "group": "x"}))
    for name in ("plot_correlation_bar", "plot_correlation_heatmap", "plot_group_heatmap", "plot_redundancy_network"):
        monkeypatch.setattr(module, name, lambda *a, **k: None)

    results = module.run_correlation_analysis(output_dir="synthetic", results_dir=None)

    assert "spearman_ts" not in results
    assert results["corr_matrix"].shape == (1, 1)
