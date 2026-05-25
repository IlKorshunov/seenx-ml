"""Shared helpers for LOO neural curve experiments (legacy import path ``train_retention_lstm_loo``)."""

from __future__ import annotations

from train.loo.common import clip01 as _clip01
from train.loo.common import curve_metrics as _curve_metrics
from train.loo.neural_curve_inputs import build_integration_matrix as _build_integration_matrix
from train.loo.neural_curve_inputs import compute_percentile_curves as _compute_percentile_curves
from train.loo.neural_curve_inputs import knn_weighted_baseline as _knn_weighted_baseline
from train.loo.neural_curve_inputs import load_features_llm_payload as _load_features_llm_payload
from train.loo.neural_curve_inputs import make_time_features as _make_time_features
from train.loo.neural_curve_inputs import resolve_training_device as _resolve_device
from train.loo.transformer.trainer_v2 import train_single_model_kw_aliases as _train_single_model
from src.normalize.curves import savgol_smooth as _savgol_smooth
from src.normalize.curves import smooth_curve_savgol_then_max_step as _smooth_postprocess
from src.normalize.curves import soft_non_increasing as _soft_non_increasing
from src.normalize.tabular import standardize_apply as _standardize_apply
from src.normalize.tabular import standardize_fit as _standardize_fit

__all__ = [
    "_build_integration_matrix",
    "_clip01",
    "_compute_percentile_curves",
    "_curve_metrics",
    "_knn_weighted_baseline",
    "_load_features_llm_payload",
    "_make_time_features",
    "_reduce_dim",
    "_resolve_device",
    "_savgol_smooth",
    "_smooth_postprocess",
    "_soft_non_increasing",
    "_standardize_apply",
    "_standardize_fit",
    "_train_single_model",
]
