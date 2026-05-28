import json
import sys
import types

import numpy as np
import pandas as pd
from tests.helpers import load_module


def test_pred_curve_run_loads_retentions_and_discovers_ids(tmp_path):
    from src.analysis.pred_curves import run

    data_dir = tmp_path / "data"
    emb_dir = tmp_path / "emb"
    out_dir = tmp_path / "out"
    for path in (data_dir / "csv_vid", data_dir / "json_vid", data_dir / "parsed_vid", emb_dir / "emb_only", out_dir):
        path.mkdir(parents=True)
    pd.DataFrame({"time_ratio": [0.0, 1.0], "audience_watch_ratio": [1.0, 0.5]}).to_csv(data_dir / "csv_vid" / "retention.csv", index=False)
    (data_dir / "json_vid" / "retention.json").write_text(json.dumps([1.0, 0.8, 0.4]), encoding="utf-8")
    (data_dir / "parsed_vid" / "retention_parsed.json").write_text(json.dumps([{"audienceWatchRatio": 90}, {"audienceWatchRatio": 60}]), encoding="utf-8")
    (out_dir / "table_features.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    retentions = run._load_retentions(data_dir)
    vids = run._discover_vids(data_dir, emb_dir, out_dir)

    assert {"csv_vid", "json_vid", "parsed_vid"} <= set(retentions)
    assert retentions["csv_vid"].shape == (run.N_POINTS,)
    assert "emb_only" in vids
    assert "table" in vids


def test_pred_curve_features_build_dataframe_with_monkeypatched_video_utils(monkeypatch, tmp_path):
    vf = types.ModuleType("src.utils.video_features")
    vf.EMBEDDING_TYPES = ["text", "audio"]  # type: ignore[attr-defined]
    vf.find_json = lambda _data_dir, vid, name: {"duration": 10, "views": 100, "likes": 5} if name == "meta.json" else {"video_features_flat": {"score": 1.0 if vid == "a" else 3.0}}  # type: ignore[attr-defined]
    vf.meta_nums = lambda meta: (meta["duration"], meta["views"], meta["likes"])  # type: ignore[attr-defined]
    vf.llm_numeric = lambda flat: {f"llm_{k}": float(v) for k, v in flat.items()}  # type: ignore[attr-defined]
    vf.tab_means = lambda _out, vid: {"tab_mean": 2.0 if vid == "a" else np.nan}  # type: ignore[attr-defined]
    vf.load_embeddings = lambda vids, _emb, mod: ([[1.0, 0.0], [0.0, 1.0]], 2) if mod == "text" else ([None, None], 0)  # type: ignore[attr-defined]
    vf.embeddings_to_matrix = lambda vecs, n, max_dim: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.utils.video_features", vf)
    features = load_module("src.analysis.pred_curves.features", "src/analysis/pred_curves/features.py")

    df = features.build_feature_df(["a", "b"], tmp_path, tmp_path, tmp_path, emb_pca_dim=3, rng=0)

    assert list(df.index) == ["a", "b"]
    assert {"log1p_duration", "llm_score", "tab_mean", "pca_text_0", "pca_audio_2"} <= set(df.columns)
    assert df["tab_mean"].isna().sum() == 0
    assert df.filter(like="pca_audio_").to_numpy().sum() == 0.0


def test_pred_curve_trainer_with_fake_catboost(monkeypatch):
    catboost = types.ModuleType("catboost")

    class FakeCatBoostRegressor:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.mean_ = 0.0
            FakeCatBoostRegressor.instances.append(self)

        def fit(self, X, y):
            self.n_features_ = X.shape[1]
            self.mean_ = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(X.shape[0], self.mean_ + X[:, 0] * 0.01)

        def get_feature_importance(self):
            return np.arange(self.n_features_, 0, -1, dtype=float)

    catboost.CatBoostRegressor = FakeCatBoostRegressor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "catboost", catboost)
    trainer = load_module("src.analysis.pred_curves.trainer", "src/analysis/pred_curves/trainer.py")

    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.0, 1.0, 0.0]})
    targets = np.array([[1.0, 2.0], [2.0, np.nan], [3.0, 4.0]])
    names = ["a", "b"]

    models = trainer.train_models(X, targets, names, iterations=2)
    preds = trainer.predict_params(models, X, names)
    loo = trainer.loo_predict(X, targets, names, iterations=2)
    fi = trainer.feature_importance(models, X.columns.tolist())

    assert set(models) == {"a", "b"}
    assert preds.shape == (3, 2)
    assert loo.shape == targets.shape
    assert set(fi.columns) == {"param", "feature", "importance"}
    assert fi.iloc[0]["feature"] == "f1"
