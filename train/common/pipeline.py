"""Shared training helpers re-exported for transformer / multimodal entrypoints."""

from train.transformer.transformer_base import (
    add_common_args,
    apply_global_calibration,
    apply_tabular_pca,
    augmentation_kwargs,
    compute_baseline_curve,
    init_run,
    load_and_filter_data,
    make_normalizer,
    apply_params,
    predict_all_videos,
    run_loo_all,
    run_video_clustering_if_requested,
    resolve_split,
    save_mae_summary,
    save_metrics_json,
    set_model_baseline,
)

__all__ = [
    "add_common_args",
    "apply_global_calibration",
    "apply_tabular_pca",
    "augmentation_kwargs",
    "compute_baseline_curve",
    "init_run",
    "load_and_filter_data",
    "make_normalizer",
    "apply_params",
    "predict_all_videos",
    "run_loo_all",
    "run_video_clustering_if_requested",
    "resolve_split",
    "save_mae_summary",
    "save_metrics_json",
    "set_model_baseline",
]
