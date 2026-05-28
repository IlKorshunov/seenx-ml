import sys
import types

import numpy as np
import pandas as pd

from tests.helpers import load_module


def _load_seq_utils(monkeypatch):
    torch = types.ModuleType("torch")
    torch.Tensor = object  # type: ignore[attr-defined]
    torch.device = lambda name="cpu": name  # type: ignore[attr-defined]
    torch.no_grad = lambda: (lambda fn: fn)  # type: ignore[attr-defined]
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    torch_utils_data.Dataset = Dataset  # type: ignore[attr-defined]
    torch_nn = types.ModuleType("torch.nn")
    torch_nn_func = types.ModuleType("torch.nn.functional")
    torch_nn_func.mse_loss = lambda *a, **k: None  # type: ignore[attr-defined]

    augmentations = types.ModuleType("analysis.augmentations")

    class RetentionAugmentation:
        @staticmethod
        def apply_random_augmentation(x, _prob):
            out = x.copy()
            out[:, 0] = 0.0
            return out

    augmentations.RetentionAugmentation = RetentionAugmentation  # type: ignore[attr-defined]

    for name, module in {
        "torch": torch,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "torch.nn": torch_nn,
        "torch.nn.functional": torch_nn_func,
        "analysis.augmentations": augmentations,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return load_module("train.common.seq_data_utils", "train/common/seq_data_utils.py")


class TestSeqFeatureFiltering:
    def test_load_video_weights_uses_meta_and_defaults(self, tmp_path, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "meta.json").write_text('{"view_count": 100, "like_count": 10, "comment_count": 2}', encoding="utf-8")

        weights = mod.load_video_weights(["a", "missing"], tmp_path)

        assert set(weights) == {"a", "missing"}
        assert all(0.25 <= value <= 4.0 for value in weights.values())

    def test_output_curve_llm_and_meta_loaders(self, tmp_path, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        out_dir = tmp_path / "output"
        snap = tmp_path / "data"
        (out_dir).mkdir()
        (snap / "v" / "transcripts").mkdir(parents=True)
        (out_dir / "v_features.csv").write_text("time,retention,x\n00:00:00,100,1\n00:00:02,80,3\n", encoding="utf-8")
        (snap / "v" / "transcripts" / "retention_parsed.json").write_text('{"status":"ok","curve_raw":[100,90,80,70,60]}', encoding="utf-8")
        (snap / "v" / "transcripts" / "features_llm.json").write_text('{"video_features_flat":{"x":1,"target__skip":2,"video_folder":"v"}}', encoding="utf-8")
        (snap / "v" / "meta.json").write_text('{"duration": 12, "view_count": 99, "upload_date": "2024-01-01T00:00:00Z"}', encoding="utf-8")

        features = mod._load_output_features("v", out_dir)
        curve = mod._load_curve_raw("v", snap)
        llm = mod._load_llm_features("v", snap)
        meta = mod._read_video_meta("v", snap)

        assert features is not None and "time_sec" in features.columns
        assert curve is not None and len(curve[0]) == 5
        assert llm == {"x": 1, "target__skip": 2, "video_folder": "v"}
        assert meta["duration"] == 12
        assert mod._extract_numeric_llm_cols(llm) == {"llm_x": 1.0}

    def test_add_video_level_features_and_broadcast_llm(self, tmp_path, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        snap = tmp_path / "data"
        (snap / "a").mkdir(parents=True)
        (snap / "b").mkdir(parents=True)
        (snap / "a" / "meta.json").write_text('{"duration": 3, "view_count": 9, "upload_date": "2024-01-01T00:00:00Z"}', encoding="utf-8")
        (snap / "b" / "meta.json").write_text('{"duration": 2, "view_count": 0, "upload_date": "2024-01-02T00:00:00Z"}', encoding="utf-8")
        video_dfs = {
            "a": pd.DataFrame({"retention": [100.0, 90.0, 80.0]}),
            "b": pd.DataFrame({"retention": [80.0, 70.0]}),
        }

        enriched = mod._add_video_level_features(video_dfs, snap)
        broadcast = mod._broadcast_llm(pd.DataFrame({"retention": [1.0, 2.0]}), {"foo": 3, "target__skip": 4, "video_folder": "x"})

        assert {"duration_sec", "log1p_view_count", "mean_retention_prior", "early_retention_drop_30s"}.issubset(enriched["a"].columns)
        assert broadcast["llm_foo"].tolist() == [3.0, 3.0]
        assert "target__skip" not in broadcast.columns

    def test_load_merged_video_combines_output_curve_and_llm(self, tmp_path, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        out_dir = tmp_path / "output"
        snap = tmp_path / "data"
        out_dir.mkdir()
        (snap / "v").mkdir(parents=True)
        (out_dir / "v_features.csv").write_text(",retention,x\n0,100,1\n1,80,3\n", encoding="utf-8")
        (snap / "v" / "retention.json").write_text('[{"audience_watch_ratio":1.0,"time_ratio":0.0},{"audience_watch_ratio":0.5,"time_ratio":1.0},{"audience_watch_ratio":0.25,"time_ratio":1.0},{"audience_watch_ratio":0.2,"time_ratio":1.0},{"audience_watch_ratio":0.1,"time_ratio":1.0}]', encoding="utf-8")
        (snap / "v" / "features_llm.json").write_text('{"video_features_flat":{"foo":2}}', encoding="utf-8")

        df = mod.load_merged_video("v", out_dir, snap, use_curve_raw=True)

        assert df is not None
        assert "llm_foo" in df.columns
        assert len(df) == 5
        np.testing.assert_allclose(df["retention"].values, [100.0, 50.0, 25.0, 20.0, 10.0])

    def test_load_snapshot_only_video(self, tmp_path, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        snap = tmp_path / "data"
        (snap / "v").mkdir(parents=True)
        (snap / "v" / "retention.json").write_text('[{"audience_watch_ratio":1.0,"time_ratio":0.0},{"audience_watch_ratio":0.8,"time_ratio":0.25},{"audience_watch_ratio":0.6,"time_ratio":0.5},{"audience_watch_ratio":0.4,"time_ratio":0.75},{"audience_watch_ratio":0.2,"time_ratio":1.0}]', encoding="utf-8")
        (snap / "v" / "features_llm.json").write_text('{"video_features_flat":{"foo":2}}', encoding="utf-8")

        df = mod._load_snapshot_only_video("v", snap)

        assert df is not None
        assert "retention" in df.columns
        assert "llm_foo" in df.columns

    def test_filter_features_drops_bad_columns_and_applies_top_k(self, tmp_path, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        results = tmp_path / "results"
        results.mkdir()
        pd.DataFrame({"avg_rank": {"keep": 1.0, "other": 2.0}}).to_csv(results / "master_ranking.csv")
        corr_dir = results / "correlation"
        corr_dir.mkdir()
        pd.DataFrame([{"feature_a": "keep", "feature_b": "redundant", "correlation": 0.95}]).to_csv(corr_dir / "redundant_pairs.csv", index=False)
        video_dfs = {
            "a": pd.DataFrame(
                {
                    "retention": [100, 90, 80],
                    "keep": [1.0, 2.0, 3.0],
                    "redundant": [1.0, 2.0, 3.0],
                    "zero": [0.0, 0.0, 0.0],
                    "sparse": [0.0, 0.0, 1.0],
                    "mostly_nan": [np.nan, np.nan, 1.0],
                    "frame": [1, 2, 3],
                }
            )
        }

        features, log = mod.filter_features(video_dfs, results_dir=results, min_nonzero_pct=0.5, max_nan_pct=0.5, top_k=1)

        assert features == ["keep"]
        assert any("Dropped zero-variance" in line for line in log)
        assert any("Dropped redundant" in line for line in log)

    def test_load_redundant_pairs_and_master_ranking_missing_files(self, tmp_path, monkeypatch):
        mod = _load_seq_utils(monkeypatch)

        assert mod._load_redundant_pairs(tmp_path) == []
        assert mod._load_master_ranking(tmp_path) == {}

    def test_loader_edge_cases_return_none_or_defaults(self, tmp_path, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        out = tmp_path / "output"
        snap = tmp_path / "snapshot"
        out.mkdir()
        (snap / "bad" / "transcripts").mkdir(parents=True)
        (snap / "bad" / "retention.json").write_text("{broken", encoding="utf-8")
        (snap / "bad" / "features_llm.json").write_text("{broken", encoding="utf-8")
        (snap / "bad" / "meta.json").write_text("[1,2]", encoding="utf-8")
        (out / "no_ret_features.csv").write_text("time,x\n00:00:00,1\n", encoding="utf-8")
        (out / "badtime_features.csv").write_text("time,retention,x\nbad,100,1\n", encoding="utf-8")

        assert mod._load_output_features("missing", out) is None
        assert mod._load_output_features("no_ret", out) is None
        assert mod._load_output_features("badtime", out) is not None
        assert mod._load_curve_raw("bad", snap) is None
        assert mod._load_llm_features("bad", snap) is None
        assert mod._read_video_meta("bad", snap) == {}
        assert mod._meta_duration_sec({}, pd.DataFrame({"time_sec": [1.0, 4.0]})) == 4.0
        assert mod._meta_duration_sec({"duration": "bad"}, pd.DataFrame({"x": [1, 2, 3]})) == 2.0
        assert mod._meta_view_count({"view_count": "bad"}) == 0.0
        assert mod._meta_published_at({}) is pd.NaT
        assert mod._load_snapshot_only_video("bad", snap) is None
        assert mod.load_merged_video("missing", out, snap) is None

    def test_load_all_merged_discovers_filters_and_snapshot_only(self, tmp_path, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        out = tmp_path / "output"
        snap = tmp_path / "snapshot"
        emb = tmp_path / "missing_embeddings"
        out.mkdir()
        for vid, duration, base_ret in [("flat", 20, 120), ("sub", 30, 80), ("snap_only", 25, 1.0)]:
            (snap / vid).mkdir(parents=True)
            (snap / vid / "meta.json").write_text(f'{{"duration": {duration}, "view_count": 10}}', encoding="utf-8")
        pd.DataFrame({"retention": [base_ret for base_ret in range(12)], "x": np.arange(12)}).to_csv(out / "flat_features.csv")
        (out / "ignored_features.csv.partial").write_text("retention\n1\n", encoding="utf-8")
        (out / "sub").mkdir()
        pd.DataFrame({"retention": np.linspace(90, 70, 12), "y": np.arange(12)}).to_csv(out / "sub" / "features_readable.csv", index=False)
        (snap / "snap_only" / "retention.json").write_text(
            str([{"audience_watch_ratio": 1.0 - i * 0.02, "time_ratio": i / 11} for i in range(12)]).replace("'", '"'),
            encoding="utf-8",
        )

        loaded = mod.load_all_merged(out, snap, use_curve_raw=False, embeddings_root=emb, emb_pca_components=2, min_duration_sec=21, max_duration_sec=29)

        assert list(loaded) == ["snap_only"]
        assert loaded["snap_only"]["duration_sec"].iloc[0] == 25

        with np.testing.assert_raises(RuntimeError):
            mod.load_all_merged(out, snap, use_curve_raw=False, embeddings_root=emb, emb_pca_components=0, min_duration_sec=999)
        with np.testing.assert_raises(RuntimeError):
            mod.load_all_merged(out, snap, use_curve_raw=False, embeddings_root=emb, emb_pca_components=0, max_duration_sec=1)
        with np.testing.assert_raises(FileNotFoundError):
            mod.load_all_merged(tmp_path / "absent", snap)


    def test_merge_embedding_pca_handles_missing_empty_and_present_embeddings(self, tmp_path, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        dfs = {
            "has": pd.DataFrame({"retention": [1.0, 2.0, 3.0]}),
            "missing": pd.DataFrame({"retention": [1.0, 2.0]}),
            "empty": pd.DataFrame({"retention": [1.0]}),
        }

        skipped = mod._merge_embedding_pca({k: v.copy() for k, v in dfs.items()}, tmp_path / "absent", n_components=2)
        assert all(list(skipped[vid].columns) == list(dfs[vid].columns) for vid in dfs)

        emb_root = tmp_path / "emb"
        emb_root.mkdir()

        def fake_load(vid, _root, duration_sec):
            if vid == "has":
                return np.ones((duration_sec, 9), dtype=np.float32), {}
            if vid == "empty":
                return np.zeros((0, 0), dtype=np.float32), {}
            raise RuntimeError("no embeddings")

        class FakePCA:
            def __init__(self, n_components):
                self.n_components = n_components

            def fit(self, _arr):
                return self

            def explained_variance_ratio(self):
                return {"vis": np.array([0.5, 0.25]), "aud": np.array([0.1]), "txt": np.array([0.05])}

            def transform(self, arr):
                return np.arange(arr.shape[0] * self.n_components * 3, dtype=np.float32).reshape(arr.shape[0], self.n_components * 3)

        monkeypatch.setattr(mod, "load_aligned_embeddings", fake_load)
        monkeypatch.setattr(mod, "PerModalityPCA", FakePCA)

        merged = mod._merge_embedding_pca({k: v.copy() for k, v in dfs.items()}, emb_root, n_components=2)

        assert "emb_vis_pc0" in merged["has"].columns
        assert merged["has"].filter(like="emb_").shape[1] == 6
        assert merged["missing"].filter(like="emb_").to_numpy().sum() == 0.0



class TestFeatureNormalizer:
    def test_fit_transform_and_retention_roundtrip(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        dfs = {
            "v": pd.DataFrame({"retention": [10.0, 50.0, 90.0], "wps": [0.0, 1.0, 3.0], "x": [1.0, 2.0, 3.0]}),
        }
        norm = mod.FeatureNormalizer()
        norm.fit(dfs, ["wps", "x"])

        x = np.array([[1.0, 2.0]], dtype=np.float32)
        transformed = norm.transform(x)
        ret_norm = norm.normalize_retention(np.array([10.0, 90.0]))
        ret_back = norm.denormalize_retention(ret_norm)

        assert transformed.shape == (1, 2)
        np.testing.assert_allclose(ret_back, [10.0, 90.0], atol=1e-5)

    def test_apply_norm_preserves_original_nan_as_zero(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        norm = mod.FeatureNormalizer()
        norm.median = np.array([0.0, 0.0])
        norm.iqr = np.array([1.0, 1.0])
        norm.log_mask = np.array([False, False])

        out = mod._apply_norm_X(np.array([[np.nan, 2.0]], dtype=np.float32), norm)

        np.testing.assert_allclose(out, [[0.0, 2.0]])


class TestSeqArrayHelpers:
    def test_tabular_array_from_df_outputs_targets_and_ad_flags(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        df = pd.DataFrame({"retention": [100.0, 90.0, 95.0], "x": [1.0, np.nan, 3.0], "is_ad": [0.0, 1.0, 0.0]})

        x, y, is_ad, spikes = mod._tabular_array_from_df(df, ["x"], normalizer=None)

        np.testing.assert_allclose(x[:, 0], [1.0, 0.0, 3.0])
        np.testing.assert_allclose(y, [100.0, 90.0, 95.0])
        np.testing.assert_allclose(is_ad, [0.0, 1.0, 0.0])
        assert spikes.shape == y.shape

    def test_window_start_indices_include_final_window(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)

        assert mod._window_start_indices(5, window_size=10, stride=3) == [0]
        assert mod._window_start_indices(10, window_size=4, stride=3) == [0, 3, 6]
        assert mod._window_start_indices(11, window_size=4, stride=3) == [0, 3, 6, 7]

    def test_align_embedding_rows_pads_truncates_and_zeros_missing(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)

        missing = mod._align_embedding_rows(None, 3, 2)
        padded = mod._align_embedding_rows(np.ones((1, 2), dtype=np.float32), 3, 2)
        truncated = mod._align_embedding_rows(np.ones((5, 2), dtype=np.float32), 3, 2)

        assert missing.shape == (3, 2)
        np.testing.assert_allclose(missing, 0.0)
        np.testing.assert_allclose(padded, [[1, 1], [0, 0], [0, 0]])
        assert truncated.shape == (3, 2)

    def test_augment_tabular_features_masks_and_adds_noise(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        x = np.ones((2, 2), dtype=np.float32)
        monkeypatch.setattr(np.random, "randn", lambda *shape: np.ones(shape, dtype=np.float32))

        out = mod._augment_tabular_features(x, feature_mask_prob=1.0, noise_std=0.5)

        np.testing.assert_allclose(out, [[0.5, 1.5], [0.5, 1.5]])


class TestSeqTimeAndResampling:
    def test_time_helpers(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        df = pd.DataFrame({"time_sec": [0.0, 2.0, 4.0]})

        assert mod.time_feature_extra_dim("none") == 0
        assert mod.time_feature_extra_dim("frac") == 1
        assert mod.time_feature_extra_dim("frac_sec") == 2
        np.testing.assert_allclose(mod._time_sec_per_row(df), [0.0, 2.0, 4.0])
        assert mod.max_time_sec_over_videos({"v": df}, ["v"]) == 4.0

    def test_resample_dataframe_interpolates_numeric_and_keeps_object_nearest(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        df = pd.DataFrame({"x": [0.0, 10.0], "video_folder": ["a", "b"]})

        out = mod.resample_dataframe_to_n_points(df, 3)

        np.testing.assert_allclose(out["x"].values, [0.0, 5.0, 10.0])
        assert out["video_folder"].tolist() == ["a", "a", "b"]

    def test_resample_dataframe_repeats_single_row(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        df = pd.DataFrame({"x": [3.0], "label": ["a"]})

        out = mod.resample_dataframe_to_n_points(df, 3)

        assert out["x"].tolist() == [3.0, 3.0, 3.0]
        assert out["label"].tolist() == ["a", "a", "a"]

    def test_resample_embeddings_to_match_dfs(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        emb = np.array([[0.0], [10.0]], dtype=np.float32)
        dfs = {"v": pd.DataFrame({"x": [1, 2, 3]})}

        out = mod.resample_embeddings_to_match_dfs({"v": emb}, dfs)

        np.testing.assert_allclose(out["v"][:, 0], [0.0, 5.0, 10.0])

    def test_append_time_features_to_matrix(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        x = np.ones((4, 1), dtype=np.float32)
        time_sec = np.array([0.0, 2.0, 4.0], dtype=np.float32)

        out = mod._append_time_features_to_matrix(x, start=0, n_full=3, ws=4, real_len=3, time_sec_full=time_sec, mode="frac_sec", ref_time_sec_max=4.0)

        assert out.shape == (4, 3)
        np.testing.assert_allclose(out[:, 1], [0.0, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(out[:, 2], [0.0, 0.5, 1.0, 1.0])


class TestSeqPredictionMetrics:
    def test_smooth_calibrate_metrics_and_ad_drop(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)

        y_true = np.array([0.0, 1.0, 2.0, 3.0])
        y_pred = np.array([0.0, 2.0, 2.0, 4.0])
        calibrated = mod.calibrate_scale(y_pred, y_true)
        metrics = mod.seq_metrics(y_pred, y_true)
        dropped = mod._apply_ad_drop(np.array([10.0, 10.0, 10.0]), np.array([0.0, 1.0, 1.0]), max_drop=2.0)

        assert calibrated.shape == y_pred.shape
        assert set(metrics) >= {"mae", "rmse", "pearson", "spearman"}
        np.testing.assert_allclose(dropped, [10.0, 8.0, 8.0])

    def test_smooth_predictions_handles_short_series(self, monkeypatch):
        mod = _load_seq_utils(monkeypatch)
        arr = np.array([1.0, 2.0, 3.0])

        np.testing.assert_allclose(mod.smooth_predictions(arr, window=15), arr)
