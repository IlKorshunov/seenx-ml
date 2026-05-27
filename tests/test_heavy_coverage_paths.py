from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def restore_real_modules():
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and not hasattr(torch_mod, "__version__"):
        for module_name in list(sys.modules):
            if module_name == "torch" or module_name.startswith("torch."):
                sys.modules.pop(module_name, None)
    tqdm_mod = sys.modules.get("tqdm")
    if tqdm_mod is not None and not hasattr(tqdm_mod, "__version__"):
        for module_name in list(sys.modules):
            if module_name == "tqdm" or module_name.startswith("tqdm."):
                sys.modules.pop(module_name, None)
    for module_name in list(sys.modules):
        if module_name == "train.common.seq_data_utils":
            sys.modules.pop(module_name, None)
        if module_name == "analysis.augmentations":
            sys.modules.pop(module_name, None)
        if module_name == "analysis.video_clustering":
            sys.modules.pop(module_name, None)
        if module_name == "src.utils.video_features":
            sys.modules.pop(module_name, None)
        if module_name == "src.utils.embedding_aligner":
            sys.modules.pop(module_name, None)


def test_retention_augmentation_datasets_and_branches(monkeypatch):
    from analysis.augmentations import AugmentedRetentionDataset, RetentionAugmentation, RetentionDataset

    np.testing.assert_allclose(RetentionAugmentation.apply_random_augmentation(np.array(3.0), 1.0), np.array(3.0))

    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: np.ones(kwargs["size"]) * 0.1)
    one_d = RetentionAugmentation.apply_random_augmentation(np.array([1.0, 2.0]), 1.0)
    assert one_d.shape == (2,)

    rand_values = iter([0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(rand_values))
    monkeypatch.setattr(np.random, "randint", lambda low, high=None: 1)
    monkeypatch.setattr(np.random, "choice", lambda n, k, replace=False: np.array([0]))
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: np.ones(kwargs["size"]) * 0.01)
    monkeypatch.setattr(np.random, "uniform", lambda low, high: 0.8)
    aug = RetentionAugmentation.apply_random_augmentation(np.arange(20, dtype=float).reshape(5, 4), 1.0)
    assert aug.shape == (5, 4)

    data = pd.DataFrame(
        {
            "video_id": ["a", "a", "b"],
            "interval_idx": [1, 0, 0],
            "target": [0.2, 0.1, 0.3],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        }
    )
    ds = RetentionDataset(data, ["x", "y"], ["a", "b"], scaler=None, fit_scaler=True, max_seq_len=4)
    assert len(ds) == 2
    assert ds[0][0].shape == (4, 2)

    monkeypatch.setattr(np.random, "rand", lambda: 1.0)
    ads = AugmentedRetentionDataset(data, ["x", "y"], ["a"], scaler=ds.scaler, fit_scaler=False, max_seq_len=4, augment=True, augment_prob=0.0, num_augmentations=2)
    assert len(ads) == 3
    assert ads[0][0].shape == (4, 2)


def test_embedding_alignment_and_video_feature_helpers(tmp_path):
    from src.utils import embedding_aligner as ea
    from src.utils import video_features as vf

    root = tmp_path / "emb"
    vid_dir = root / "v1"
    vid_dir.mkdir(parents=True)
    np.save(vid_dir / "visual_embeddings.npy", np.arange(9, dtype=np.float32).reshape(3, 3))
    np.save(vid_dir / "audio_embeddings.npy", np.ones((2, 3), dtype=np.float32))
    np.save(vid_dir / "seg_embeddings.npy", np.eye(2, 3, dtype=np.float32))
    (vid_dir / "seg_meta.json").write_text(json.dumps([{"start": 0, "end": 1.5}, {"start": 1, "end": 3}]), encoding="utf-8")

    visual = ea.resample_visual_to_1fps(np.ones((1, 3), dtype=np.float32), 3)
    audio = ea.resample_audio_to_1fps(np.ones((0, 2), dtype=np.float32), 3)
    text = ea.resample_text_to_1fps(np.eye(2, 3, dtype=np.float32), [{"start": 0, "end": 2}, {"start": 1, "end": 3}], 3)
    aligned, duration = ea.load_aligned_embeddings("v1", root, duration_sec=3, modalities=("visual", "audio", "text"))
    align_features = ea.get_alignment_features("v1", root, duration_sec=3)

    assert visual.shape == (3, 3)
    assert audio.shape == (3, 2)
    assert text.shape == (3, 3)
    assert aligned.shape == (3, 9)
    assert duration == 3
    assert align_features is not None and align_features.shape == (3, 3)
    np.testing.assert_allclose(ea.normalize_l2(np.array([[3.0, 4.0]])).sum(), 1.4)

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    (data_dir / "v1" / "transcripts").mkdir(parents=True)
    out_dir.mkdir()
    (data_dir / "v1" / "transcripts" / "meta.json").write_text('{"statistics":{"viewCount": "7"}, "likeCount": 3, "duration": 5}', encoding="utf-8")
    (data_dir / "v1" / "features_llm.json").write_text('{"video_features_flat":{"foo":1,"bar":true,"target__bad":2}}', encoding="utf-8")
    pd.DataFrame({"brightness": [1.0, 3.0], "wps": [2.0, 4.0], "skip": ["x", "y"]}).to_csv(out_dir / "v1_features.csv")
    np.save(vid_dir / "bert_embeddings.npy", np.arange(4, dtype=np.float32).reshape(2, 2))

    meta = vf.find_json(data_dir, "v1", "meta.json")
    assert vf.meta_nums(meta) == (5.0, 7.0, 3.0)
    assert vf.llm_numeric({"foo": 1, "flag": True, "target__x": 2, "video_folder": "v"}) == {"llm_foo": 1.0, "llm_flag": 1.0}
    assert vf.tab_means(out_dir, "v1") == {"tab_brightness": 2.0, "tab_wps": 3.0}
    vecs, max_dim = vf.load_embeddings(["v1", "missing"], root, "bert")
    mat = vf.embeddings_to_matrix(vecs, 2, max_dim)
    reduced = vf.pca_reduce(mat, 3, 42)
    assert reduced.shape == (2, 3)


def test_seq_data_utils_real_torch_datasets_losses_and_prediction(tmp_path):
    import torch
    import train.common.seq_data_utils as su

    df = pd.DataFrame(
        {
            "retention": [100.0, 90.0, 85.0, 88.0, 80.0, 75.0],
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "wps": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "is_ad": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "time_sec": [0, 2, 4, 6, 8, 10],
        }
    )
    video_dfs = {"v": df}
    norm = su.FeatureNormalizer()
    norm.fit(video_dfs, ["x", "wps"])

    ds = su.WindowedSeqDataset(video_dfs, ["v"], ["x", "wps"], normalizer=norm, window_size=4, stride=3, video_weights={"v": 1.5}, time_feature_mode="frac_sec", ref_time_sec_max=10)
    assert len(ds) == 2
    item = ds[0]
    assert item["features"].shape == (4, 4)
    assert float(item["video_weight"]) == 1.5

    padded_ds = su.WindowedSeqDataset({"v": df.iloc[:2]}, ["v"], ["x", "wps"], window_size=5)
    assert padded_ds[0]["padding_mask"].tolist() == [False, False, True, True, True]

    emb = np.ones((3, 5), dtype=np.float32)
    mds = su.MultimodalWindowedDataset(video_dfs, {"v": emb}, ["v"], ["x", "wps"], normalizer=norm, window_size=4, stride=3, emb_dim=5, time_feature_mode="frac")
    mitem = mds[0]
    assert mitem["embeddings"].shape == (4, 5)
    assert mitem["tabular"].shape[1] == 3

    pred = torch.tensor([[1.0, 0.8, 0.7, 0.9, 0.6]], dtype=torch.float32)
    target = torch.tensor([[1.0, 0.9, 0.75, 0.8, 0.7]], dtype=torch.float32)
    mask = torch.tensor([[False, False, False, False, True]])
    is_ad = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0]])
    spikes = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    loss = su.composite_loss(pred, target, is_ad, spikes, mask, video_weight=torch.tensor([2.0]))
    assert torch.isfinite(loss)
    assert torch.isfinite(su.ad_aware_loss(pred, target, is_ad, spikes, mask, torch.nn.MSELoss()))

    class TabModel(torch.nn.Module):
        def forward(self, x):
            return x[..., 0] * 0.0 + 0.5

    y_true, y_pred = su.predict_video(TabModel(), df.iloc[:3], ["x", "wps"], norm, torch.device("cpu"), window_size=5, apply_smoothing=True, time_feature_mode="frac")
    assert y_true.shape == y_pred.shape == (3,)

    class MMModel(torch.nn.Module):
        def forward(self, embeddings, tabular=None):
            return embeddings[..., 0] * 0.0 + tabular[..., 0] * 0.0 + 0.25

    y_true_mm, y_pred_mm = su.predict_video_multimodal(MMModel(), df, emb, ["x", "wps"], norm, torch.device("cpu"), window_size=4, stride=2, emb_dim=5)
    assert y_true_mm.shape == y_pred_mm.shape == (6,)

    out_dir = tmp_path / "plots"
    su.plot_mae_summary({"v": {"mae": 1.0, "rmse": 2.0, "split": "val", "n_seconds": 3}}, str(out_dir), model_name="T")
    assert (out_dir / "mae_summary.png").is_file()
    assert (out_dir / "mae_summary.csv").is_file()


def test_video_clustering_synthetic_pipeline_outputs(tmp_path, monkeypatch):
    import analysis.video_clustering as vc

    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(0, 0.05, (4, 3)), rng.normal(3, 0.05, (4, 3))])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

    assert vc._k_range(2, 5, 4) == [2, 3]
    assert vc._n_unique(labels) == 2
    assert vc.cluster_kmeans(X, 2, 3, 42)[1] in {2, 3}
    assert vc.cluster_gmm(X, 2, 3, 42)[1] in {2, 3}
    assert vc.cluster_dbscan(X, min_samples=2)[0].shape == (8,)
    assert vc.cluster_spectral(X, 2, 3, 42)[0].shape == (8,)
    D = vc.pairwise_distances(X)
    assert vc.cluster_agglo(D, 2, 3)[0].shape == (8,)

    metrics_x = vc.compute_metrics(labels, X=X)
    metrics_d = vc.compute_metrics(labels, D=D)
    entropy = vc.compute_entropy_external(labels, np.array([0, 0, 1, 1, 0, 0, 1, 1]))
    assert metrics_x["k"] == 2 and metrics_d["k"] == 2
    assert "H_class|clust" in entropy

    monkeypatch.setattr(vc, "project_2d", lambda arr, rng, precomputed=False: np.column_stack([np.arange(arr.shape[0]), np.arange(arr.shape[0])]))
    out = tmp_path / "cluster"
    out.mkdir()
    vids = [f"v{i}" for i in range(8)]
    display = [{"video_id": vid, "duration_sec": i + 1, "view_count": i * 10, "like_count": i} for i, vid in enumerate(vids)]
    vc.save_results(vids, labels, np.column_stack([np.arange(8), np.arange(8)]), display, "kmeans", metrics_x, out)
    vc.save_comparison({"kmeans": metrics_x, "dtw": metrics_d}, out)
    assert (out / "clusters.json").is_file()
    assert (out / "comparison.csv").is_file()

    data_dir = tmp_path / "data"
    emb_dir = tmp_path / "emb"
    output_dir = tmp_path / "output"
    for base in (data_dir, emb_dir, output_dir):
        base.mkdir()
    for idx, vid in enumerate(["a", "b", "c"]):
        (data_dir / vid).mkdir()
        (emb_dir / vid).mkdir()
        (data_dir / vid / "meta.json").write_text(json.dumps({"duration": 10 + idx, "view_count": 100 + idx, "like_count": 5 + idx}), encoding="utf-8")
        (data_dir / vid / "features_llm.json").write_text(json.dumps({"video_features_flat": {"pace": idx + 1}}), encoding="utf-8")
        pd.DataFrame({"brightness": [idx, idx + 1], "target": [1, 2]}).to_csv(output_dir / f"{vid}_features.csv", index=False)
        for modality in vc.EMBEDDING_TYPES:
            np.save(emb_dir / vid / f"{modality}_embeddings.npy", np.ones((2, 3), dtype=np.float32) * (idx + 1))

    ids = vc.discover_video_ids(data_dir, emb_dir, output_dir)
    full_df, feature_cols = vc.load_all_video_sequences(ids, output_dir)
    mat, names, display_rows = vc.build_feature_matrix(ids, data_dir, emb_dir, output_dir, emb_pca_dim=2, rng=42)

    assert ids == ["a", "b", "c"]
    assert not full_df.empty and "brightness" in feature_cols
    assert mat.shape[0] == 3
    assert len(names) == mat.shape[1]
    assert len(display_rows) == 3
