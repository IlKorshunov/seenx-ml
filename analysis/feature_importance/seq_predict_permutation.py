"""Loss-based feature importance for models exposed via ``predict_video``."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from train.common.seq_data_utils import predict_video

from train.common.retention_plots import COLOR_ACTUAL as C_BLUE, GRID_ALPHA, save_figure as _save_fig

import logging
from typing import cast

logger = logging.getLogger(__name__)


def _predict_video_metrics(model, feature_cols, video_dfs, video_ids, normalizer, device, window_size, predict_kwargs: dict | None = None) -> dict[str, dict[str, float]]:
    predict_kwargs = predict_kwargs or {}
    metrics = {}
    for video_id in video_ids:
        y_true, y_pred = predict_video(model, video_dfs[video_id], feature_cols, normalizer, device, window_size, **predict_kwargs)
        error = y_pred - y_true
        metrics[video_id] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
        }
    return metrics


def _mean_metric(metrics_by_video: dict[str, dict[str, float]], metric: str) -> float:
    return float(np.mean([m[metric] for m in metrics_by_video.values()]))


def _plot_importance(df: pd.DataFrame, score_col: str, title: str, out_path: str, top_n: int) -> None:
    top = df.sort_values(score_col, ascending=False).head(top_n)
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.3)))
    ax.barh(top["feature"][::-1], top[score_col][::-1], color=C_BLUE)
    ax.set(xlabel=score_col.replace("_", " "), title=title)
    ax.grid(True, alpha=GRID_ALPHA, axis="x")
    plt.tight_layout()
    _save_fig(fig, out_path)


def _build_master_ranking(method_frames: dict[str, pd.DataFrame], out_dir: str) -> pd.DataFrame:
    ranks: list[pd.Series] = []
    for method_name, df in method_frames.items():
        ranked = cast(pd.Series, df.set_index("feature")["importance_mae_delta"].rank(ascending=False))
        ranked.name = f"rank_{method_name}"
        ranks.append(ranked)

    if not ranks:
        master = pd.DataFrame(columns=["feature", "avg_rank", "n_methods"])
    else:
        merged = pd.concat(ranks, axis=1)
        merged["avg_rank"] = merged.mean(axis=1)
        merged["n_methods"] = merged.notna().sum(axis=1) - 1
        master = merged.sort_values("avg_rank").reset_index()

    master.to_csv(os.path.join(out_dir, "master_ranking.csv"), index=False)
    if not master.empty:
        plot_df = master.assign(importance_mae_delta=master["avg_rank"].max() - master["avg_rank"] + 1)
        _plot_importance(plot_df, "importance_mae_delta", "Seq Loss Feature Importance Consensus", os.path.join(out_dir, "master_ranking.png"), min(30, len(master)))
    return master


def _write_summary(out_dir: str, baseline_mae: float, baseline_rmse: float, method_frames: dict[str, pd.DataFrame]) -> None:
    ls = os.linesep
    lines = [
        f"Seq retention feature importance{ls}",
        f"Baseline MAE: {baseline_mae:.6f}{ls}",
        f"Baseline RMSE: {baseline_rmse:.6f}{ls}",
        f"{ls}Methods:{ls}",
    ]
    for method_name, df in method_frames.items():
        lines.append(f"- {method_name}: {len(df)} features, score = metric_after_perturbation - baseline_metric{ls}")
    with open(os.path.join(out_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
        f.writelines(lines)


def compute_predict_video_loss_importance(model, feature_cols, video_dfs, val_ids, normalizer, device, out_dir, window_size, n_repeats=5, top_n=30, predict_kwargs: dict | None = None):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    baseline_metrics = _predict_video_metrics(model, feature_cols, video_dfs, val_ids, normalizer, device, window_size, predict_kwargs)
    baseline_mae = _mean_metric(baseline_metrics, "mae")
    baseline_rmse = _mean_metric(baseline_metrics, "rmse")
    pd.DataFrame([{"video": video_id, **metrics} for video_id, metrics in baseline_metrics.items()]).to_csv(os.path.join(out_dir, "baseline_metrics.csv"), index=False)

    rng = np.random.RandomState(42)
    method_frames = {}

    permutation_rows = []
    for feat_idx, feat_name in enumerate(tqdm(feature_cols)):
        mae_deltas, rmse_deltas = [], []
        for _ in range(n_repeats):
            perturbed = {}
            for video_id in val_ids:
                shuffled_df = video_dfs[video_id].copy()
                shuffled_df[feat_name] = rng.permutation(shuffled_df[feat_name].values)
                perturbed[video_id] = shuffled_df
            metrics = _predict_video_metrics(model, feature_cols, perturbed, val_ids, normalizer, device, window_size, predict_kwargs)
            mae_deltas.append(_mean_metric(metrics, "mae") - baseline_mae)
            rmse_deltas.append(_mean_metric(metrics, "rmse") - baseline_rmse)
        permutation_rows.append(
            {
                "feature": feat_name,
                "feature_idx": feat_idx,
                "importance_mae_delta": float(np.mean(mae_deltas)),
                "importance_rmse_delta": float(np.mean(rmse_deltas)),
                "importance_mae_std": float(np.std(mae_deltas)),
                "importance_rmse_std": float(np.std(rmse_deltas)),
                "method": "permutation",
            }
        )
    permutation_df = pd.DataFrame(permutation_rows).sort_values("importance_mae_delta", ascending=False).reset_index(drop=True)
    permutation_df.to_csv(os.path.join(out_dir, "permutation_importance.csv"), index=False)
    permutation_df.rename(columns={"importance_mae_delta": "importance_mae_increase"}).to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
    _plot_importance(permutation_df, "importance_mae_delta", f"Seq Permutation Importance by MAE Delta (top {top_n})", os.path.join(out_dir, "permutation_importance.png"), top_n)
    method_frames["permutation"] = permutation_df

    median_values = getattr(normalizer, "median", None)
    ablation_rows = []
    for feat_idx, feat_name in enumerate(tqdm(feature_cols, desc="Seq loss median ablation importance")):
        fill_value = float(median_values[feat_idx]) if median_values is not None else float(np.nanmedian(np.concatenate([video_dfs[video_id][feat_name].values for video_id in val_ids])))
        perturbed = {}
        for video_id in val_ids:
            ablated_df = video_dfs[video_id].copy()
            ablated_df[feat_name] = fill_value
            perturbed[video_id] = ablated_df
        metrics = _predict_video_metrics(model, feature_cols, perturbed, val_ids, normalizer, device, window_size, predict_kwargs)
        ablation_rows.append(
            {
                "feature": feat_name,
                "feature_idx": feat_idx,
                "importance_mae_delta": _mean_metric(metrics, "mae") - baseline_mae,
                "importance_rmse_delta": _mean_metric(metrics, "rmse") - baseline_rmse,
                "fill_value": fill_value,
                "method": "median_ablation",
            }
        )
    ablation_df = pd.DataFrame(ablation_rows).sort_values("importance_mae_delta", ascending=False).reset_index(drop=True)
    ablation_df.to_csv(os.path.join(out_dir, "median_ablation_importance.csv"), index=False)
    _plot_importance(ablation_df, "importance_mae_delta", f"Seq Median Ablation Importance by MAE Delta (top {top_n})", os.path.join(out_dir, "median_ablation_importance.png"), top_n)
    method_frames["median_ablation"] = ablation_df

    master = _build_master_ranking(method_frames, out_dir)
    _write_summary(out_dir, baseline_mae, baseline_rmse, method_frames)
    logger.info("Saved seq loss feature importance (%d features, baseline MAE=%.4f, RMSE=%.4f)", len(feature_cols), baseline_mae, baseline_rmse)
    return {"baseline": baseline_metrics, "permutation": permutation_df, "median_ablation": ablation_df, "master_ranking": master}


def compute_predict_video_permutation_importance(model, feature_cols, video_dfs, val_ids, normalizer, device, out_dir, window_size, n_repeats=5):
    result = compute_predict_video_loss_importance(model, feature_cols, video_dfs, val_ids, normalizer, device, out_dir, window_size, n_repeats=n_repeats)
    result["permutation"].rename(columns={"importance_mae_delta": "importance_mae_increase"}).to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
    return result
