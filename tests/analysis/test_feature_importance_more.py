"""Synthetic tests for feature-importance analysis modules."""

import sys
import types
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _toy_xy():
    X = pd.DataFrame({"rms": [0.0, 1.0, 2.0, 3.0], "edit_pace": [3.0, 2.0, 1.0, 0.0], "unknown_x": [1.0, 1.0, 2.0, 2.0]})
    y = pd.Series([0.0, 1.0, 2.0, 3.0])
    return X, y


class TestPermutationImportance:
    def test_importance_frame_sorts_and_groups_features(self):
        from analysis.feature_importance import permutation_importance as perm

        out = perm._importance_frame(["edit_pace", "rms", "mystery"], np.array([0.2, 0.9, -0.1]), np.array([0.01, 0.02, 0.03]), "fake")

        assert out["feature"].tolist() == ["rms", "edit_pace", "mystery"]
        assert out.loc[out["feature"] == "rms", "group"].iloc[0] == "audio_basic"
        assert out.loc[out["feature"] == "mystery", "group"].iloc[0] == "unknown"

    def test_compute_permutation_importance_uses_model_and_sklearn_result(self, monkeypatch):
        from analysis.feature_importance import permutation_importance as perm

        X, y = _toy_xy()

        class Model:
            def fit(self, X_arr, y_arr):
                self.shape = X_arr.shape
                return self

        monkeypatch.setitem(perm.MODEL_BUILDERS, "fake", lambda _n: Model())

        def fake_sklearn_perm(model, X_arr, y_arr, **kwargs):
            assert model.shape == X_arr.shape
            return types.SimpleNamespace(importances_mean=np.array([0.3, 0.1, 0.2]), importances_std=np.array([0.03, 0.01, 0.02]))

        monkeypatch.setattr(perm, "sklearn_perm", fake_sklearn_perm)
        out = perm.compute_permutation_importance(X, y, model_name="fake", n_repeats=2)

        assert out["feature"].tolist() == ["rms", "unknown_x", "edit_pace"]

    def test_loo_falls_back_for_tiny_dataset(self, monkeypatch):
        from analysis.feature_importance import permutation_importance as perm

        X, y = _toy_xy()
        called = {}
        monkeypatch.setattr(perm, "compute_permutation_importance", lambda *args, **kwargs: called.setdefault("out", pd.DataFrame({"feature": ["rms"]})))

        out = perm.compute_loo_permutation_importance(X.iloc[:3], y.iloc[:3], model_name="fake")

        assert out.equals(called["out"])

    def test_loo_permutation_aggregates_fold_importances(self, monkeypatch):
        from analysis.feature_importance import permutation_importance as perm

        X = pd.DataFrame({"rms": [0, 1, 2, 3, 4], "edit_pace": [4, 3, 2, 1, 0]})
        y = pd.Series([0, 1, 2, 3, 4])

        class Model:
            def fit(self, X_arr, y_arr):
                return self

        monkeypatch.setitem(perm.MODEL_BUILDERS, "fake_loo", lambda _n: Model())
        monkeypatch.setattr(
            perm,
            "sklearn_perm",
            lambda *args, **kwargs: types.SimpleNamespace(importances_mean=np.array([0.2, 0.4])),
        )

        out = perm.compute_loo_permutation_importance(X, y, model_name="fake_loo", n_repeats=1)

        assert out["feature"].tolist() == ["edit_pace", "rms"]
        assert out["model"].unique().tolist() == ["fake_loo_loo"]

    def test_run_permutation_importance_orchestrates_models(self, tmp_path, monkeypatch):
        from analysis.feature_importance import permutation_importance as perm

        X, y = _toy_xy()
        monkeypatch.setattr(perm, "load_all_videos", lambda output_dir: {"v": pd.DataFrame()})
        monkeypatch.setattr(perm, "aggregate_per_video", lambda video_dfs: pd.DataFrame({"dummy": [1]}))
        monkeypatch.setattr(perm, "prepare_X_y", lambda agg_df, target: (X, y))
        monkeypatch.setattr(perm, "compute_permutation_importance", lambda X, y, model_name, **kwargs: perm._importance_frame(X.columns.tolist(), np.arange(len(X.columns)), np.zeros(len(X.columns)), model_name))
        monkeypatch.setattr(perm, "plot_permutation_importance", lambda *args, **kwargs: None)
        monkeypatch.setattr(perm, "plot_multi_model_comparison", lambda *args, **kwargs: None)

        out = perm.run_permutation_importance(output_dir="unused", results_dir=str(tmp_path), use_loo=False, models=["ridge", "rf"], top_n=2)

        assert set(out) == {"ridge", "rf"}
        assert (tmp_path / "permutation" / "perm_importance_ridge.csv").exists()


class TestCatBoostImportance:
    def test_builtin_and_shap_importance_with_fake_model(self, monkeypatch):
        from analysis.feature_importance import catboost_importance as cb

        X, y = _toy_xy()

        class FakePool:
            def __init__(self, X_arr, y_arr=None):
                self.X_arr = X_arr
                self.y_arr = y_arr

        class FakeModel:
            def fit(self, X_arr, y_arr):
                self.n_features = X_arr.shape[1]
                return self

            def get_feature_importance(self, pool, type):
                if type == "ShapValues":
                    return np.array([[1.0, -2.0, 0.5, 0.1], [2.0, -1.0, 0.0, 0.1], [3.0, -0.5, 1.0, 0.1], [4.0, -0.25, 1.5, 0.1]])
                return np.array([0.2, 0.8, 0.1])

        monkeypatch.setattr(cb, "Pool", FakePool)
        monkeypatch.setattr(cb, "_build_catboost", lambda: FakeModel())

        builtin = cb.compute_builtin_importance(X, y)
        shap_df, shap_matrix = cb.compute_shap_importance(X, y)

        assert builtin["feature"].iloc[0] == "edit_pace"
        assert shap_matrix.shape == (4, 3)
        assert shap_df["feature"].iloc[0] == "rms"

    def test_group_importance_consensus_and_run_step(self, tmp_path, monkeypatch):
        from analysis.feature_importance import catboost_importance as cb

        frame = pd.DataFrame({"feature": ["rms", "edit_pace"], "importance": [2.0, 1.0], "group": ["audio_basic", "visual_motion"]})

        group = cb.compute_group_importance(frame)
        consensus = cb._build_consensus(frame, frame.iloc[::-1].reset_index(drop=True), frame)

        assert group["importance_pct"].sum() == pytest.approx(100.0)
        assert set(consensus.columns) == {"feature", "rank_pvc", "rank_lfc", "rank_shap", "avg_rank", "group"}

        plots = []
        monkeypatch.setattr(cb, "save_importance_csv", lambda df, path, sort_by=None: df.to_csv(path, index=False))
        out = cb._run_importance_step({}, "x", "message", lambda: frame, tmp_path / "imp.csv", lambda df: plots.append(df))

        assert out.equals(frame)
        assert plots == [frame]
        assert (tmp_path / "imp.csv").exists()

    def test_plot_helpers_and_run_catboost_importance(self, tmp_path, monkeypatch):
        from analysis.feature_importance import catboost_importance as cb

        X, y = _toy_xy()
        frame = pd.DataFrame({"feature": ["rms", "edit_pace", "unknown_x"], "importance": [3.0, 2.0, 1.0], "group": ["audio_basic", "visual_motion", "unknown"], "method": ["m"] * 3})
        shap_matrix = np.arange(12, dtype=float).reshape(4, 3)

        cb.plot_top_features(frame, "top", tmp_path / "top.png", top_n=3, color_by_group=True)
        cb.plot_top_features(frame, "top", tmp_path / "top_plain.png", top_n=2, color_by_group=False)
        cb.plot_group_importance(cb.compute_group_importance(frame), "groups", tmp_path / "groups.png")
        cb.plot_shap_beeswarm(X, shap_matrix, tmp_path / "beeswarm.png", top_n=2)

        assert (tmp_path / "top.png").exists()
        assert (tmp_path / "top_plain.png").exists()
        assert (tmp_path / "groups.png").exists()
        assert (tmp_path / "beeswarm.png").exists()

        monkeypatch.setattr(cb, "load_all_videos", lambda output_dir: {"v": pd.DataFrame()})
        monkeypatch.setattr(cb, "aggregate_per_video", lambda video_dfs: pd.DataFrame({"dummy": [1]}))
        monkeypatch.setattr(cb, "prepare_X_y", lambda agg_df, target: (X, y))
        monkeypatch.setattr(cb, "compute_builtin_importance", lambda X, y, importance_type: frame.assign(method=importance_type))
        monkeypatch.setattr(cb, "compute_shap_importance", lambda X, y: (frame.assign(method="SHAP_mean_abs"), shap_matrix))
        monkeypatch.setattr(cb, "plot_top_features", lambda *args, **kwargs: None)
        monkeypatch.setattr(cb, "plot_group_importance", lambda *args, **kwargs: None)
        monkeypatch.setattr(cb, "plot_shap_beeswarm", lambda *args, **kwargs: None)

        result = cb.run_catboost_importance(output_dir="unused", results_dir=str(tmp_path), top_n=2)

        assert {"pvc", "lfc", "shap", "group", "consensus"} <= set(result)
        assert (tmp_path / "catboost" / "importance_consensus.csv").exists()


class TestShapAnalysis:
    def test_shap_value_helpers_and_group_bar(self, tmp_path, monkeypatch):
        from analysis.feature_importance import shap_analysis as shap_mod

        X, _ = _toy_xy()

        class FakePool:
            def __init__(self, X_arr):
                self.X_arr = X_arr

        class FakeModel:
            def get_feature_importance(self, pool, type):
                return np.column_stack([np.ones((len(pool.X_arr), X.shape[1])), np.full(len(pool.X_arr), 0.5)])

        monkeypatch.setattr(shap_mod, "Pool", FakePool)
        matrix, expected = shap_mod.compute_shap_values_catboost(FakeModel(), X)

        assert matrix.shape == (4, 3)
        assert expected == 0.5

        out_path = tmp_path / "group.png"
        shap_mod.plot_shap_group_bar(matrix, X, out_path)
        assert out_path.exists()

    def test_fallback_plots_and_dependence_missing_feature(self, tmp_path):
        from analysis.feature_importance import shap_analysis as shap_mod

        X, _ = _toy_xy()
        matrix = np.arange(12, dtype=float).reshape(4, 3)

        shap_mod._fallback_shap_bar(matrix, X, tmp_path / "bar.png", top_n=2)
        shap_mod._fallback_dependence(matrix, X, "rms", tmp_path / "dep.png")
        shap_mod._fallback_dependence(matrix, X, "missing", tmp_path / "missing.png")

        assert (tmp_path / "bar.png").exists()
        assert (tmp_path / "dep.png").exists()
        assert not (tmp_path / "missing.png").exists()

    def test_sklearn_shap_and_plot_wrappers(self, tmp_path, monkeypatch):
        from analysis.feature_importance import shap_analysis as shap_mod

        X, _ = _toy_xy()

        class Model:
            def predict(self, X_arr):
                return X_arr.sum(axis=1)

        class FakeExplainer:
            expected_value = 1.25

            def shap_values(self, X_arr, nsamples=100):
                return np.ones_like(X_arr, dtype=float)

        monkeypatch.setattr(shap_mod.shap, "sample", lambda X_arr, n: X_arr[:n])
        monkeypatch.setattr(shap_mod.shap, "KernelExplainer", lambda predict, background: FakeExplainer())

        values, expected = shap_mod.compute_shap_values_sklearn(Model(), X, background_samples=2)

        assert values.shape == X.shape
        assert expected == 1.25

        monkeypatch.setattr(shap_mod.shap, "summary_plot", lambda *args, **kwargs: None)
        monkeypatch.setattr(shap_mod.shap, "waterfall_plot", lambda *args, **kwargs: None)
        monkeypatch.setattr(shap_mod.shap, "dependence_plot", lambda *args, **kwargs: None)
        monkeypatch.setattr(shap_mod.shap, "force_plot", lambda *args, **kwargs: "<html>force</html>")
        monkeypatch.setattr(shap_mod.shap, "save_html", lambda path, html: Path(path).write_text(str(html), encoding="utf-8"))

        shap_mod.plot_shap_summary(values, X, tmp_path / "summary.png", max_display=2, plot_type="bar")
        shap_mod.plot_shap_waterfall(values, X, expected, 0, tmp_path / "waterfall.png", title="sample")
        shap_mod.plot_shap_dependence(values, X, "rms", tmp_path / "dep_real.png")
        shap_mod.save_shap_html(values, X, expected, tmp_path / "force.html")

        assert (tmp_path / "summary.png").exists()
        assert (tmp_path / "waterfall.png").exists()
        assert (tmp_path / "dep_real.png").exists()
        assert (tmp_path / "force.html").exists()

    def test_run_shap_analysis_orchestrates_outputs(self, tmp_path, monkeypatch):
        from analysis.feature_importance import shap_analysis as shap_mod

        X, y = _toy_xy()
        matrix = np.arange(12, dtype=float).reshape(4, 3)

        class Model:
            def predict(self, X_arr):
                return X_arr[:, 0]

        monkeypatch.setattr(shap_mod, "load_all_videos", lambda output_dir: {"v": pd.DataFrame()})
        monkeypatch.setattr(shap_mod, "aggregate_per_video", lambda video_dfs: pd.DataFrame({"dummy": [1]}))
        monkeypatch.setattr(shap_mod, "prepare_X_y", lambda agg_df, target: (X, y))
        monkeypatch.setattr(shap_mod, "_train_catboost", lambda X, y: Model())
        monkeypatch.setattr(shap_mod, "compute_shap_values_catboost", lambda model, X: (matrix, 0.5))
        for name in ("plot_shap_summary", "plot_shap_group_bar", "plot_shap_waterfall", "save_shap_html", "plot_shap_dependence"):
            monkeypatch.setattr(shap_mod, name, lambda *args, **kwargs: None)

        result = shap_mod.run_shap_analysis(output_dir="unused", results_dir=str(tmp_path), top_n=2, n_dependence_plots=2)

        assert result["shap_matrix"].shape == matrix.shape
        assert (tmp_path / "shap" / "shap_values.csv").exists()


class TestRunAllFeatureImportance:
    def test_build_master_ranking_collects_all_pipeline_shapes(self, tmp_path):
        run_all = importlib.import_module("analysis.feature_importance.run_all")

        base = pd.DataFrame({"feature": ["rms", "edit_pace"], "importance": [2.0, 1.0]})
        perm = pd.DataFrame({"feature": ["rms", "unknown_x"], "importance_mean": [0.4, 0.8]})
        corr = pd.DataFrame({"feature": ["edit_pace"], "abs_correlation": [0.9]})
        mi = pd.DataFrame({"feature": ["rms"], "mutual_information": [0.7]})
        shap = pd.DataFrame({"feature": ["unknown_x"], "shap_mean_abs": [1.2]})

        result = run_all.build_master_ranking(
            {
                "catboost": {"pvc": base, "lfc": base, "shap": base},
                "permutation": {"ridge": perm},
                "correlation": {"spearman_agg": corr, "pearson_agg": corr, "mutual_info": mi},
                "shap": {"importance_df": shap},
                "transformer": {"attn_importance": base, "grad_importance": base},
            },
            tmp_path,
        )

        assert set(result["feature"]) == {"rms", "edit_pace", "unknown_x"}
        assert (tmp_path / "master_ranking.csv").exists()
        assert result["avg_rank"].notna().all()

    def test_summary_and_plot_outputs(self, tmp_path):
        run_all = importlib.import_module("analysis.feature_importance.run_all")

        ranking = pd.DataFrame({"feature": ["rms", "edit_pace"], "group": ["audio_basic", "visual_motion"], "avg_rank": [1.0, 2.0], "n_methods": [3, 2]})

        (tmp_path / "catboost").mkdir()
        (tmp_path / "catboost" / "importance.csv").write_text("feature,importance\nrms,1\n", encoding="utf-8")
        run_all.write_summary_report(ranking, {"catboost": {}}, tmp_path, "target_avg_retention", {"catboost": 0.1})
        run_all.plot_master_ranking(ranking, tmp_path, top_n=2)

        assert (tmp_path / "summary_report.md").exists()
        assert (tmp_path / "master_ranking.png").exists()


class TestSeqPredictPermutation:
    def _install_train_stubs(self, monkeypatch):
        train = types.ModuleType("train")
        common_pkg = types.ModuleType("train.common")
        train.__path__ = []  # type: ignore[attr-defined]
        common_pkg.__path__ = []  # type: ignore[attr-defined]
        seq = types.ModuleType("train.common.seq_data_utils")
        plots = types.ModuleType("train.common.retention_plots")
        seq.predict_video = lambda model, df, feature_cols, normalizer, device, window_size, **kwargs: (df["target"].to_numpy(float), df[feature_cols].sum(axis=1).to_numpy(float))
        plots.COLOR_ACTUAL = "#1565C0"
        plots.GRID_ALPHA = 0.25
        plots.save_figure = lambda fig, out_path: fig.savefig(out_path)
        for name, module in {
            "train": train,
            "train.common": common_pkg,
            "train.common.seq_data_utils": seq,
            "train.common.retention_plots": plots,
        }.items():
            monkeypatch.setitem(sys.modules, name, module)

    def test_loss_importance_writes_tables_and_master_ranking(self, tmp_path, monkeypatch):
        self._install_train_stubs(monkeypatch)
        from tests.helpers import load_module

        seq_imp = load_module("analysis.feature_importance.seq_predict_permutation", "analysis/feature_importance/seq_predict_permutation.py")

        class Model:
            def eval(self):
                return None

        normalizer = types.SimpleNamespace(median=np.array([0.5, 0.5]))
        video_dfs = {
            "v1": pd.DataFrame({"rms": [0.0, 1.0, 2.0], "edit_pace": [2.0, 1.0, 0.0], "target": [0.0, 1.0, 2.0]}),
            "v2": pd.DataFrame({"rms": [1.0, 2.0, 3.0], "edit_pace": [0.0, 1.0, 2.0], "target": [1.0, 2.0, 3.0]}),
        }

        out = seq_imp.compute_predict_video_loss_importance(Model(), ["rms", "edit_pace"], video_dfs, ["v1", "v2"], normalizer, "cpu", str(tmp_path), 2, n_repeats=1)

        assert {"baseline", "permutation", "median_ablation", "master_ranking"} <= set(out)
        assert (tmp_path / "baseline_metrics.csv").exists()
        assert (tmp_path / "master_ranking.csv").exists()


class TestTransformerImportance:
    def test_small_transformer_attention_gradient_and_plots(self, tmp_path):
        from analysis.feature_importance import transformer_importance as tr

        X, y = _toy_xy()
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.float32)
        model, losses = tr._train_with_tracking(X_arr, y_arr, d_model=8, n_heads=2, n_layers=1, epochs=2, batch_size=2, device="cpu")

        attn, matrix = tr.extract_attention_importance(model, X_arr, "cpu")
        grad = tr.extract_gradient_importance(model, X_arr, y_arr, "cpu")

        assert attn.shape == (3,)
        assert matrix.shape == (3, 3)
        assert grad.shape == (3,)
        assert len(losses) == 2

        attn_df = pd.DataFrame({"feature": X.columns, "importance": attn, "group": ["audio_basic", "visual_motion", "unknown"]})
        grad_df = pd.DataFrame({"feature": X.columns, "importance": grad, "group": ["audio_basic", "visual_motion", "unknown"]})
        tr.plot_attention_importance(attn_df, "attn", tmp_path / "attn.png", top_n=3)
        tr.plot_attention_heatmap(matrix, X.columns.tolist(), tmp_path / "heat.png", top_n=3)
        tr.plot_combined_importance(attn_df, grad_df, tmp_path / "combined.png", top_n=3)
        tr.plot_training_curve(losses, tmp_path / "loss.png")

        assert (tmp_path / "attn.png").exists()
        assert (tmp_path / "heat.png").exists()
        assert (tmp_path / "combined.png").exists()
        assert (tmp_path / "loss.png").exists()
