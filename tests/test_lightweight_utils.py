import json
from argparse import Namespace

import numpy as np
import pytest

from src.utils import embedding_aligner as aligner
from train.common.split_utils import apply_train_id_file_filter, resolve_train_val_split
from train.common.tuned_params_io import apply_best_params_to_args, load_tuned_json, merge_tuned_file_into_args


class TestEmbeddingAligner:
    def test_resample_visual_interpolates_to_duration(self):
        embs = np.array([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32)

        out = aligner.resample_visual_to_1fps(embs, duration_sec=3)

        np.testing.assert_allclose(out, [[0.0, 0.0], [5.0, 10.0], [10.0, 20.0]])
        assert out.dtype == np.float32

    def test_resample_audio_uses_chunk_midpoints(self):
        embs = np.array([[0.0], [10.0]], dtype=np.float32)

        out = aligner.resample_audio_to_1fps(embs, duration_sec=3)

        np.testing.assert_allclose(out[:, 0], [0.0, 10.0, 10.0])

    def test_resample_text_weights_overlapping_segments(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        seg_meta = [{"start": 0.0, "end": 2.0}, {"start": 1.0, "end": 3.0}]

        out = aligner.resample_text_to_1fps(embs, seg_meta, duration_sec=3)

        np.testing.assert_allclose(out[0], [1.0, 0.0])
        np.testing.assert_allclose(out[1], [0.5, 0.5])
        np.testing.assert_allclose(out[2], [0.0, 1.0])

    def test_load_aligned_embeddings_concatenates_requested_modalities(self, tmp_path):
        root = tmp_path / "embeddings" / "vid"
        root.mkdir(parents=True)
        np.save(root / "visual_embeddings.npy", np.ones((2, 3), dtype=np.float32))
        np.save(root / "audio_embeddings.npy", np.full((2, 2), 2.0, dtype=np.float32))

        out, duration = aligner.load_aligned_embeddings("vid", embeddings_root=tmp_path / "embeddings", duration_sec=2, modalities=("visual", "audio"))

        assert duration == 2
        assert out.shape == (2, 5)
        np.testing.assert_allclose(out[:, :3], 1.0)
        np.testing.assert_allclose(out[:, 3:], 2.0)

    def test_load_aligned_embeddings_fills_missing_modalities_with_zeros(self, tmp_path):
        out, duration = aligner.load_aligned_embeddings("missing", embeddings_root=tmp_path, duration_sec=4, modalities=("text",))

        assert duration == 4
        assert out.shape == (4, aligner.TEXT_DIM)
        assert not out.any()

    def test_normalize_l2_and_cosine_between_modalities(self):
        vis = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        aud = np.array([[3.0, 4.0], [0.0, 1.0]], dtype=np.float32)
        txt = np.array([[4.0, -3.0], [1.0, 0.0]], dtype=np.float32)

        normed = aligner.normalize_l2(vis)
        np.testing.assert_allclose(np.linalg.norm(normed, axis=1), [1.0, 1.0], rtol=1e-6)

        cos = aligner.cosine_between_modalities(vis, aud, txt)
        np.testing.assert_allclose(cos[0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(cos[1], [0.0, 1.0, 0.0], atol=1e-6)


class TestSplitUtils:
    def test_eval_video_takes_priority(self):
        args = Namespace(eval_video="b", val_first_n_output=2, random_seed=0, val_ratio=0.5)

        train_ids, val_ids = resolve_train_val_split(args, ["a", "b", "c"], ["a", "b", "c"])

        assert train_ids == ["a", "c"]
        assert val_ids == ["b"]

    def test_val_first_n_output_uses_output_order(self):
        args = Namespace(eval_video="", val_first_n_output=2, random_seed=0, val_ratio=0.5)

        train_ids, val_ids = resolve_train_val_split(args, ["a", "b", "c", "d"], ["d", "b", "a", "c"])

        assert val_ids == ["d", "b"]
        assert train_ids == ["a", "c"]

    def test_random_split_is_seeded_and_non_empty(self):
        args = Namespace(eval_video="", val_first_n_output=0, random_seed=42, val_ratio=0.25)

        train_ids, val_ids = resolve_train_val_split(args, ["a", "b", "c", "d"], ["a", "b", "c", "d"])

        assert len(val_ids) == 1
        assert sorted(train_ids + val_ids) == ["a", "b", "c", "d"]

    def test_apply_train_id_file_filter(self, tmp_path):
        allow_file = tmp_path / "ids.txt"
        allow_file.write_text("b\nc\n\n", encoding="utf-8")
        args = Namespace(train_video_ids_file=str(allow_file))

        assert apply_train_id_file_filter(["a", "b", "c"], args) == ["b", "c"]


class TestTunedParamsIo:
    def test_load_tuned_json_validates_file_and_object(self, tmp_path):
        payload = tmp_path / "best.json"
        payload.write_text(json.dumps({"best_params": {"lr": 0.1}}), encoding="utf-8")

        assert load_tuned_json(payload) == {"best_params": {"lr": 0.1}}

        missing = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError):
            load_tuned_json(missing)

        not_object = tmp_path / "list.json"
        not_object.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError):
            load_tuned_json(not_object)

    def test_apply_best_params_updates_shared_fields_and_stride(self):
        args = Namespace()

        result = apply_best_params_to_args(
            args,
            {"lr": 0.01, "window_size": 9, "top_k_features": 25},
            model_family="multimodal_transformer",
            apply_architecture=False,
        )

        assert result.lr == 0.01
        assert result.window_size == 9
        assert result.window_stride == 4
        assert result.tuned_top_k == 25
        assert not hasattr(result, "d_model")

    def test_apply_best_params_can_apply_architecture(self):
        args = Namespace()

        result = apply_best_params_to_args(
            args,
            {"d_model": 128, "n_heads": 4, "n_layers": 2, "dropout": 0.2},
            model_family="multimodal_transformer",
            apply_architecture=True,
        )

        assert result.d_model == 128
        assert result.n_heads == 4
        assert result.n_layers == 2
        assert result.dropout == 0.2

    def test_merge_tuned_file_into_args_applies_and_saves_copy(self, tmp_path):
        source = tmp_path / "best.json"
        source.write_text(json.dumps({"best_params": {"hidden_size": 64, "n_layers": 3}}), encoding="utf-8")
        copy_to = tmp_path / "out" / "copy.json"
        args = Namespace()

        result = merge_tuned_file_into_args(args, source, model_family="multimodal_lstm", apply_architecture=True, save_copy_to=copy_to)

        assert result.d_model == 64
        assert result.n_layers == 3
        assert json.loads(copy_to.read_text(encoding="utf-8")) == {"best_params": {"hidden_size": 64, "n_layers": 3}}

    def test_merge_tuned_file_requires_best_params_object(self, tmp_path):
        source = tmp_path / "bad.json"
        source.write_text(json.dumps({"best_params": []}), encoding="utf-8")

        with pytest.raises(ValueError):
            merge_tuned_file_into_args(Namespace(), source, model_family="multimodal_lstm", apply_architecture=True)
