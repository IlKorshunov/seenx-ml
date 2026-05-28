from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def restore_real_torch_and_module():
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and not hasattr(torch_mod, "FloatTensor"):
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                sys.modules.pop(name, None)
    sys.modules.pop("analysis.embedding_clustering", None)
    sys.modules.pop("analysis.augmentations", None)


def _module():
    return pytest.importorskip("analysis.embedding_clustering")


def test_positional_retention_transformer_and_sampling_helpers():
    torch = pytest.importorskip("torch")
    ec = _module()

    pe = ec.PositionalEncoding(d_model=6, dropout=0.0, max_len=8)
    assert pe(torch.zeros(4, 1, 6)).shape == (4, 1, 6)

    model = ec.RetentionTransformer(input_dim=3, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0)
    x = torch.randn(2, 5, 3)
    out = model(x)
    pred, emb = model(x, return_embeddings=True)
    assert out.shape == (2, 1)
    assert pred.shape == (2, 1)
    assert emb.shape == (2, 8)

    sampled_1d = ec._sample_embedding_sequence(np.arange(4, dtype=np.float32), n_steps=3)
    sampled_2d = ec._sample_embedding_sequence(np.arange(12, dtype=np.float32).reshape(4, 3), n_steps=5)
    assert sampled_1d.shape == (3, 4)
    assert sampled_2d.shape == (5, 3)
    assert ec._pca_reduce(np.ones((2, 3), dtype=np.float32), pca_dim=0).shape == (2, 3)


def test_precomputed_embedding_blocks_and_files(tmp_path):
    ec = _module()
    emb_root = tmp_path / "emb"
    for vid, scale in [("a", 1.0), ("b", 2.0), ("c", 3.0)]:
        vdir = emb_root / vid
        vdir.mkdir(parents=True)
        np.save(vdir / "audio_embeddings.npy", np.ones((2, 3), dtype=np.float32) * scale)
        if vid != "c":
            np.save(vdir / "visual_embeddings.npy", np.eye(3, dtype=np.float32) * scale)

    matrix, vids = ec.extract_precomputed_embeddings(["a", "b", "c", "missing"], emb_root, n_steps=4, pca_dim=2)

    assert vids == ["a", "b", "c"]
    assert matrix.shape == (3, 2)

    seqs = {
        "a": {"audio_embeddings.npy": np.ones((2, 3), dtype=np.float32)},
        "b": {"visual_embeddings.npy": np.ones((2, 2), dtype=np.float32)},
    }
    blocks = ec._scaled_modality_blocks(seqs, ["a", "b"], {"audio_embeddings.npy": 3, "visual_embeddings.npy": 2}, n_steps=2)
    assert len(blocks) == 2
    assert blocks[0].shape == (2, 6)


def test_extract_video_embeddings_and_optimal_cluster_outputs(tmp_path):
    torch = pytest.importorskip("torch")
    ec = _module()
    old_steps = ec.N_STEPS
    ec.N_STEPS = 4
    try:
        df = pd.DataFrame(
            {
                "video_id": ["b", "a", "a", "b"],
                "interval_idx": [0, 1, 0, 1],
                "x": [0.0, 1.0, 2.0, 3.0],
                "target": [0.0, 1.0, 2.0, 3.0],
            }
        )

        class Scaler:
            def transform(self, x):
                return x

        class Model(torch.nn.Module):
            def forward(self, features, return_embeddings=False):
                emb = features.mean(dim=1)
                pred = emb.sum(dim=1, keepdim=True)
                return (pred, emb) if return_embeddings else pred

        embeddings, video_ids = ec.extract_video_embeddings(df, ["x"], Scaler(), Model(), torch.device("cpu"))
    finally:
        ec.N_STEPS = old_steps

    assert video_ids == ["a", "b"]
    assert embeddings.shape == (2, 1)

    clustered = np.vstack([np.random.default_rng(1).normal(0, 0.01, (4, 2)), np.random.default_rng(2).normal(3, 0.01, (4, 2))])
    k = ec.find_optimal_clusters(clustered, min_clusters=2, max_clusters=3, output_dir=tmp_path)
    assert k in {2, 3}
    assert (tmp_path / "optimal_clusters.html").exists()
    assert (tmp_path / "optimal_clusters.png").exists()
    assert ec.find_optimal_clusters(np.ones((2, 2)), min_clusters=3, max_clusters=4) == 3


def test_cluster_aware_transformer_training_and_prediction(monkeypatch):
    torch = pytest.importorskip("torch")
    ec = _module()
    cat = ec.ClusterAwareTransformer(
        n_clusters=2,
        input_dim=2,
        device="cpu",
        model_config={"d_model": 8, "nhead": 2, "num_layers": 1, "dim_feedforward": 16, "dropout": 0.0},
    )
    labels = cat.fit_clusters(np.array([[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]]))
    assert sorted(set(labels.tolist())) == [0, 1]
    cat.cluster_assignments = {"a": 0, "b": 0, "c": 1}

    df = pd.DataFrame(
        {
            "video_id": ["a", "a", "b", "b", "c", "c"],
            "interval_idx": [0, 1, 0, 1, 0, 1],
            "x": [0.0, 0.1, 0.2, 0.3, 4.0, 4.1],
            "y": [1.0, 1.1, 1.2, 1.3, 5.0, 5.1],
            "target": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1],
        }
    )

    class Scaler:
        def transform(self, x):
            return x

    model, loss = cat.train_for_cluster(0, ["a", "b"], ["c"], df, ["x", "y"], Scaler(), epochs=1, patience=1, use_augmentation=False)
    assert model is not None
    assert np.isfinite(loss)
    assert cat.train_for_cluster(1, ["c"], ["a"], df, ["x", "y"], Scaler(), epochs=1)[0] is None

    pred = cat.predict("a", torch.zeros(1, 4, 2))
    assert pred.shape == (1,)
