from __future__ import annotations

import torch


def safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = p.clamp(float(eps), 1.0 - float(eps))
    return torch.log(x / (1.0 - x))


class TemporalConvBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_sizes: list[int], dropout: float):
        super().__init__()
        nb = len(kernel_sizes)
        bc = max(1, channels // nb)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(channels, bc, k, padding=k // 2) for k in kernel_sizes])
        tot = bc * nb
        self.proj = torch.nn.Linear(tot, channels) if tot != channels else torch.nn.Identity()
        self.norm = torch.nn.LayerNorm(channels)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        cat = torch.cat([c(xt) for c in self.convs], dim=1).transpose(1, 2)
        return self.norm(x + self.drop(self.proj(cat)))


class FeatureGate(torch.nn.Module):
    def __init__(self, fd: int, td: int):
        super().__init__()
        self.g = torch.nn.Sequential(torch.nn.Linear(td, fd * 2), torch.nn.GELU(), torch.nn.Linear(fd * 2, fd), torch.nn.Sigmoid())

    def forward(self, f: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return f * self.g(t)


class RetentionCurveTransformerV2(torch.nn.Module):

    def __init__(
        self,
        input_size,
        d_model=64,
        n_layers=3,
        n_heads=4,
        ffn_mult=2,
        dropout=0.15,
        residual_scale=1.25,
        curve_points=50,
        conv_kernels=None,
        static_feature_dim=0,
        time_ctx_dim=10,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.static_feature_dim = static_feature_dim
        self.time_ctx_dim = time_ctx_dim

        self.feature_gate = FeatureGate(static_feature_dim, time_ctx_dim) if static_feature_dim > 0 and time_ctx_dim > 0 else None

        self.input_proj = torch.nn.Sequential(torch.nn.Linear(input_size, d_model), torch.nn.LayerNorm(d_model), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.pos_embed = torch.nn.Parameter(torch.randn(1, curve_points, d_model) * 0.02)

        self.pre_conv = TemporalConvBlock(d_model, conv_kernels, dropout) if conv_kernels else None

        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ffn_mult, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.post_conv = TemporalConvBlock(d_model, conv_kernels, dropout) if conv_kernels else None

        self.head = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(d_model, 1))
        self.ad_head = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(d_model, 1))

    def forward(self, seq_inputs, baseline_curve, integration_strength):
        x = seq_inputs
        if self.feature_gate is not None and self.static_feature_dim > 0:
            sd, td = self.static_feature_dim, self.time_ctx_dim
            x = torch.cat([self.feature_gate(x[:, :, :sd], x[:, :, sd : sd + td]), x[:, :, sd:]], dim=-1)

        h = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        if self.pre_conv is not None:
            h = self.pre_conv(h)
        h = self.final_norm(self.encoder(h))
        if self.post_conv is not None:
            h = self.post_conv(h)

        raw_res = self.head(h).squeeze(-1)
        smooth_res = self.residual_scale * torch.tanh(raw_res)
        ad_drop = torch.nn.functional.softplus(self.ad_head(h).squeeze(-1))
        residual = smooth_res - integration_strength * ad_drop

        bs, steps, _ = seq_inputs.shape
        bl_logit = safe_logit(baseline_curve)[None, :].expand(bs, steps)
        return torch.sigmoid(bl_logit + residual), residual, ad_drop


RetentionTransformerV2 = RetentionCurveTransformerV2
