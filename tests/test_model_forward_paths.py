from __future__ import annotations

import types

import numpy as np
import pytest
import torch


def test_model_base_blocks_and_tabular_gate():
    from src.models import model_base as mb

    x = torch.randn(2, 5, 6)
    mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, False]])

    attn = mb.MultiHeadTemporalAttention(hidden_dim=6, n_heads=2, dropout=0.0)
    assert attn(x).shape == x.shape

    assert mb.ModalityProjection(3, 4, dropout=0.0)(torch.randn(2, 5, 3)).shape == (2, 5, 4)
    assert mb.GatedFeatureProjection(3, 4, dropout=0.0)(torch.randn(2, 5, 3)).shape == (2, 5, 4)
    assert mb.SinusoidalPositionalEncoding(6, max_len=8, dropout=0.0)(x).shape == x.shape

    proj, gate = mb.build_tabular_gate(2, 6, dropout=0.0)
    gated = mb.apply_tabular_gate(x, torch.randn(2, 5, 2), proj, gate)
    assert gated.shape == x.shape
    assert mb.apply_tabular_gate(x, None, proj, gate) is x

    lstm = torch.nn.LSTM(6, 3, batch_first=True, bidirectional=True)
    assert mb.lstm_forward_packed(lstm, x, None, total_length=5).shape == x.shape
    assert mb.lstm_forward_packed(lstm, x, mask, total_length=5).shape == x.shape

    baseline = torch.tensor([10.0, 20.0])
    deviation = torch.ones(1, 4)
    np.testing.assert_allclose(mb.apply_baseline(deviation, baseline, 4).numpy(), [[11.0, 21.0, 21.0, 21.0]])

    emb = torch.randn(1, 2, mb.VISUAL_DIM + mb.AUDIO_DIM + mb.TEXT_DIM)
    vis, aud, txt = mb.split_embeddings(emb)
    assert vis.shape[-1] == mb.VISUAL_DIM
    assert aud.shape[-1] == mb.AUDIO_DIM
    assert txt.shape[-1] == mb.TEXT_DIM


@pytest.mark.parametrize("head_type", ["tanh", "sigmoid", "cumulative"])
def test_unimodal_retention_models_forward(head_type):
    from src.models.retention_lstm import RetentionLSTM
    from src.models.retention_transformer import RetentionTransformer

    x = torch.randn(2, 6, 4)
    mask = torch.tensor([[False, False, False, False, True, True], [False, False, False, False, False, False]])

    lstm = RetentionLSTM(n_features=4, hidden_size=8, n_layers=1, dropout=0.0, bidirectional=True, n_attn_heads=2, head_type=head_type)
    transformer = RetentionTransformer(n_features=4, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0, head_type=head_type)
    if head_type == "tanh":
        lstm.set_baseline(torch.linspace(90, 80, 4))
        transformer.set_baseline(torch.linspace(90, 80, 4))

    assert lstm(x, src_key_padding_mask=mask).shape == (2, 6)
    assert transformer(x, src_key_padding_mask=mask).shape == (2, 6)


def test_multimodal_retention_models_forward_with_tabular_and_convs():
    from src.models import model_base as mb
    from src.models.retention_multimodal_lstm import MultimodalRetentionLSTM, PreConvBlock as LstmPreConv
    from src.models.retention_multimodal_transformer import MultimodalRetentionTransformer, PreConvBlock as TransformerPreConv

    emb = torch.randn(2, 5, mb.VISUAL_DIM + mb.AUDIO_DIM + mb.TEXT_DIM)
    tab = torch.randn(2, 5, 2)
    mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, False]])

    assert LstmPreConv(6, dropout=0.0)(torch.randn(2, 5, 6)).shape == (2, 5, 6)
    assert TransformerPreConv(6, dropout=0.0)(torch.randn(2, 5, 6)).shape == (2, 5, 6)

    lstm = MultimodalRetentionLSTM(
        hidden_size=8,
        n_layers=1,
        dropout=0.0,
        bidirectional=True,
        n_tabular_features=2,
        n_attn_heads=2,
        use_conv_blocks=True,
    )
    lstm.set_baseline(torch.tensor([80.0, 79.0]))
    assert lstm(emb, tabular=tab, src_key_padding_mask=mask).shape == (2, 5)

    transformer = MultimodalRetentionTransformer(
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
        n_tabular_features=2,
        use_conv_blocks=True,
    )
    transformer.set_baseline(torch.tensor([80.0, 79.0, 78.0]))
    assert transformer(emb, tabular=tab, src_key_padding_mask=mask).shape == (2, 5)

    no_mod = MultimodalRetentionTransformer(
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
        use_modality_embeddings=False,
    )
    assert no_mod(emb).shape == (2, 5)


class TinyBackbone(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
        if pixel_values is None and torch.is_tensor(input_ids) and input_ids.dtype.is_floating_point and input_ids.ndim > 2:
            pixel_values, input_ids = input_ids, None
        if pixel_values is not None:
            batch = pixel_values.shape[0]
            hidden = pixel_values.reshape(batch, pixel_values.shape[1], -1)[..., : self.config.hidden_size]
            if hidden.shape[-1] < self.config.hidden_size:
                hidden = torch.nn.functional.pad(hidden, (0, self.config.hidden_size - hidden.shape[-1]))
            return types.SimpleNamespace(last_hidden_state=self.proj(hidden.float()))
        one_hot = torch.nn.functional.one_hot(input_ids % self.config.hidden_size, num_classes=self.config.hidden_size).float()
        return types.SimpleNamespace(last_hidden_state=self.proj(one_hot))


class TinyTokenizer:
    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        batch = len(texts)
        ids = torch.arange(1, 5).repeat(batch, 1)
        mask = torch.ones_like(ids)

        class Enc(dict):
            def to(self, device):
                return Enc({k: v.to(device) for k, v in self.items()})

        return Enc({"input_ids": ids, "attention_mask": mask})


def test_bert_helpers_and_retention_heads(monkeypatch):
    from src.models import bert_retention as br

    monkeypatch.setattr(br, "_load_backbone", lambda backbone, device: (TinyBackbone(hidden_size=8).to(device), TinyTokenizer(), 8))
    monkeypatch.setattr(br, "_apply_lora", lambda model, **kwargs: model)

    pe = br.SinusoidalPE(8, max_len=8, dropout=0.0)
    assert pe(torch.zeros(1, 4, 8)).shape == (1, 4, 8)

    head = br.TemporalRegressionHead(d_model=8, n_heads=2, n_layers=1, d_ff=12, dropout=0.0)
    assert head(torch.randn(2, 4, 8), mask=torch.zeros(2, 4, dtype=torch.bool)).shape == (2, 4)

    embs = br._encode_segments(["a", "b"], TinyBackbone(8), TinyTokenizer(), torch.device("cpu"), batch_size=1)
    aligned = br._align_segments_to_1fps(embs, [{"start": 0, "end": 1.5}, {"start": 1, "end": 3}], duration_sec=3)
    assert aligned.shape == (3, 8)

    extractor = br.BERTFeatureExtractor(device=torch.device("cpu"))
    assert extractor.extract([], [], duration_sec=2).shape == (2, 8)
    assert extractor.extract(["hello"], [{"start": 0, "end": 2}], duration_sec=2, batch_size=1).shape == (2, 8)
    assert extractor(torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.long)).shape == (1, 3, 8)

    retention = br.BERTRetention(n_head_layers=1, d_ff=12, dropout=0.0)
    retention.set_baseline(torch.tensor([50.0, 49.0]))
    assert retention(torch.randn(2, 4, 8), padding_mask=torch.zeros(2, 4, dtype=torch.bool)).shape == (2, 4)
    assert retention.trainable_parameters()

    hybrid = br.BERTHybridRetention(d_model=8, n_heads=2, n_layers=1, d_ff=12, dropout=0.0, n_tabular_features=2, visual_dim=3, audio_dim=4)
    hybrid.set_baseline(torch.tensor([60.0, 59.0]))
    out = hybrid(torch.randn(2, 4, 8), torch.randn(2, 4, 3), torch.randn(2, 4, 4), tabular=torch.randn(2, 4, 2))
    assert out.shape == (2, 4)
    assert hybrid.trainable_parameters()


def test_videomae_heads_with_tiny_backbone(monkeypatch):
    from src.models import video_mae_retention as vm

    class TinyProcessor:
        def __call__(self, frames, return_tensors="pt"):
            return {"pixel_values": torch.ones(1, 16, 1, 1, 8)}

    monkeypatch.setattr(vm, "_load_backbone", lambda backbone, device: (TinyBackbone(hidden_size=8).to(device), TinyProcessor(), 8))
    monkeypatch.setattr(vm, "_apply_lora", lambda model, **kwargs: model)

    head = vm.TemporalRegressionHead(d_model=8, n_heads=2, n_layers=1, d_ff=12, dropout=0.0)
    assert head(torch.randn(1, 4, 8)).shape == (1, 4)
    assert vm._interp_to_seconds(torch.randn(1, 3, 8), 5).shape == (1, 5, 8)

    extractor = vm.VideoMAEFeatureExtractor(backbone="tiny", device=torch.device("cpu"))
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
    assert extractor.extract(frames, clip_stride=2).shape == (3, 8)
    assert extractor(torch.ones(1, 16, 1, 1, 8)).shape[0] == 1

    retention = vm.VideoMAERetention(backbone="tiny", n_head_layers=1, d_ff=12, dropout=0.0)
    retention.set_baseline(torch.tensor([70.0, 69.0]))
    assert retention(torch.ones(1, 16, 1, 1, 8), n_seconds=4).shape == (1, 4)
    assert retention.trainable_parameters()

    hybrid = vm.VideoMAEHybridRetention(backbone="tiny", d_model=8, n_heads=2, n_layers=1, d_ff=12, dropout=0.0, n_tabular_features=2, audio_dim=3, text_dim=4)
    hybrid.set_baseline(torch.tensor([80.0, 79.0]))
    assert hybrid.encode_video(torch.ones(1, 16, 1, 1, 8), n_seconds=4).shape == (1, 4, 8)
    out = hybrid(torch.randn(2, 4, 8), torch.randn(2, 4, 3), torch.randn(2, 4, 4), tabular=torch.randn(2, 4, 2))
    assert out.shape == (2, 4)
    assert hybrid.trainable_parameters()
