from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from train.common.retention_plots import plot_prediction, plot_retention_prediction
from train.common.seq_data_utils import *
from train.common.split_utils import apply_train_id_file_filter, resolve_train_val_split
from train.common.tuned_params_io import merge_tuned_file_into_args
from analysis.feature_importance.seq_predict_permutation import compute_predict_video_loss_importance
logger = logging.getLogger(__name__)
from analysis.feature_importance.run_all import PIPELINE_REGISTRY, run_all

TABULAR_FEATURE_IMPORTANCE_TARGET = "target_avg_retention"


def build_tuned_feature_filter_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    filter_kw: dict[str, Any] = {"top_k": args.top_k_features or None}
    if hasattr(args, "tuned_corr_threshold"): filter_kw["redundant_corr_threshold"] = args.tuned_corr_threshold
    if hasattr(args, "tuned_nan_pct"): filter_kw["max_nan_pct"] = args.tuned_nan_pct
    if hasattr(args, "tuned_nonzero_pct"): filter_kw["min_nonzero_pct"] = args.tuned_nonzero_pct
    return filter_kw


def resolve_output_video_ids(output_dir_features: str, video_dfs: dict) -> list[str]:
    return sorted(path.name.replace("_features.csv", "") for path in Path(output_dir_features).glob("*_features.csv") if not path.name.endswith(".partial") and path.name.replace("_features.csv", "") in video_dfs)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    a = parser.add_argument
    a("--output-dir-features", default="output")
    a("--snapshot-dir", default="data")
    a("--use-curve-raw", action="store_true", default=True)
    a("--no-use-curve-raw", dest="use_curve_raw", action="store_false")
    a("--output-dir", default="")
    a("--train-video-ids-file", default="")
    a("--eval-video", default="")
    a("--loo-all", action="store_true", default=False)
    a("--val-ratio", type=float, default=0.15)
    a("--val-first-n-output", type=int, default=0)

    a("--top-k-features", type=int, default=0)
    a("--ad-penalty-weight", type=float, default=15.0)
    a("--alpha-corr", type=float, default=0.3)
    a("--alpha-smooth", type=float, default=0.15)
    a("--alpha-delta", type=float, default=0.4)
    a("--alpha-mono", type=float, default=0.03)
    a("--start-boost-secs", type=int, default=15)
    a("--start-boost-factor", type=float, default=2.0)
    a("--engagement-weight", action="store_true", default=True)
    a("--no-engagement-weight", dest="engagement_weight", action="store_false")

    a("--window-size", type=int, default=128)
    a("--window-stride", type=int, default=64)

    a("--d-model", type=int, default=128)
    a("--n-heads", type=int, default=4)
    a("--n-layers", type=int, default=4)
    a("--d-ff", type=int, default=256)
    a("--dropout", type=float, default=0.2)

    a("--epochs", type=int, default=200)
    a("--batch-size", type=int, default=16)
    a("--lr", type=float, default=5e-4)
    a("--weight-decay", type=float, default=1e-3)
    a("--patience", type=int, default=30)
    a("--grad-clip", type=float, default=1.0)
    a("--warmup-epochs", type=int, default=10)
    a("--swa-start-epoch", type=int, default=0)
    a("--swa-lr", type=float, default=1e-4)

    a("--feature-mask-prob", type=float, default=0.1)
    a("--noise-std", type=float, default=0.02)
    a("--use-augmentation", action="store_true", default=False)

    a("--random-seed", type=int, default=42)
    a("--device", default="cpu")
    a("--apply-smoothing", action="store_true", default=False)
    a("--smooth-window", type=int, default=7)
    a("--feature-importance-top-n", type=int, default=30)
    a("--run-video-clustering", action="store_true", default=False)
    a("--video-clustering-strategy", default="all")
    a("--video-clustering-min-k", type=int, default=2)
    a("--video-clustering-max-k", type=int, default=8)
    a("--video-clustering-output-dir", default="analysis/video_clustering")
    a("--tuned-params-json", default="")
    a("--tuned-apply-architecture", action="store_true")
    a("--curve-points", type=int, default=0)
    a("--time-features", choices=["none", "frac", "frac_sec"], default="none")
    a("--use-tabular-pca", action="store_true", default=False)
    a("--tabular-pca-dim", type=int, default=64)
    a("--min-duration-sec", type=float, default=0)
    a("--max-duration-sec", type=float, default=0)
    return parser


def init_run(args: argparse.Namespace, default_output_dir: str = "") -> torch.device:
    if not args.output_dir and default_output_dir: args.output_dir = default_output_dir
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    return torch.device(args.device)


def load_and_filter_data(args: argparse.Namespace, *, extra_load_kwargs: dict | None = None, log_subpath: str = "feature_filter_log.txt") -> tuple[dict, list[str], list[str], list[str]]:
    logger.info("Loading merged data")
    load_kwargs = {"use_curve_raw": args.use_curve_raw}
    if extra_load_kwargs: load_kwargs.update(extra_load_kwargs)
    video_dfs = load_all_merged(args.output_dir_features, args.snapshot_dir, **load_kwargs)
    if getattr(args, "curve_points", 0) and args.curve_points > 0:
        video_dfs = resample_video_dfs_to_curve_points(video_dfs, args.curve_points)

    video_ids = sorted(video_dfs.keys())
    output_video_ids = resolve_output_video_ids(args.output_dir_features, video_dfs)
    filter_kw = build_tuned_feature_filter_kwargs(args)
    feature_cols, filter_log = filter_features(video_dfs, **filter_kw)
    Path(os.path.join(args.output_dir, log_subpath)).write_text("\n".join(filter_log), encoding="utf-8")
    logger.info("Features after filtering: %d", len(feature_cols))
    return video_dfs, video_ids, output_video_ids, feature_cols


def resolve_split(args: argparse.Namespace, video_ids: list[str], output_video_ids: list[str]) -> tuple[list[str], list[str]]:
    train_ids, val_ids = resolve_train_val_split(args, video_ids, output_video_ids)
    train_ids = apply_train_id_file_filter(train_ids, args)
    logger.info("Train: %d videos, Val: %d videos", len(train_ids), len(val_ids))
    return train_ids, val_ids


def make_normalizer(args: argparse.Namespace, video_dfs: dict, train_ids: list[str], feature_cols: list[str]) -> tuple[FeatureNormalizer, float, dict | None]:
    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)
    ref_sec = max_time_sec_over_videos(video_dfs, train_ids)
    logger.info("Time ref (max time_sec on train): %.1f s", ref_sec)
    video_weights = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None
    return normalizer, ref_sec, video_weights


def apply_tabular_pca(args: argparse.Namespace, video_dfs: dict[str, pd.DataFrame], train_ids: list[str], feature_cols: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    if not args.use_tabular_pca or not feature_cols:
        return video_dfs, feature_cols
    train_x = np.vstack([pd.DataFrame(video_dfs[v]).reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float) for v in train_ids])
    scaler = StandardScaler().fit(train_x)
    dim = min(max(1, int(args.tabular_pca_dim)), train_x.shape[0], train_x.shape[1])
    pca = PCA(n_components=dim, random_state=args.random_seed).fit(scaler.transform(train_x))
    cols = [f"pca_feature_{i:03d}" for i in range(dim)]
    for vid, df in video_dfs.items():
        x = df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        video_dfs[vid] = pd.concat([df.drop(columns=[c for c in feature_cols if c in df.columns]), pd.DataFrame(pca.transform(scaler.transform(x)).astype(np.float32), columns=cols, index=df.index)], axis=1)
    return video_dfs, cols


def compute_baseline_curve(video_dfs: dict, train_ids: list[str], normalizer: FeatureNormalizer) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(len(video_dfs[v]) for v in train_ids)
    acc = np.zeros(max_len, dtype=np.float64)
    cnt = np.zeros(max_len, dtype=np.float64)
    for v in train_ids:
        ret = pd.to_numeric(video_dfs[v]["retention"], errors="coerce").fillna(0).values
        acc[: len(ret)] += ret
        cnt[: len(ret)] += 1.0
    raw = (acc / np.maximum(cnt, 1.0)).astype(np.float32)
    norm = normalizer.normalize_retention(raw).astype(np.float32)
    return raw, norm


def set_model_baseline(model, video_dfs: dict, train_ids: list[str], normalizer: FeatureNormalizer) -> None:
    raw, norm = compute_baseline_curve(video_dfs, train_ids, normalizer)
    model.set_baseline(torch.tensor(norm))
    logger.info("Baseline curve set: %d points, raw_mean=%.1f%%, norm_mean=%.4f", len(raw), raw.mean(), norm.mean())

def predict_all_videos(*, video_ids: list[str], val_ids: list[str], video_dfs: dict, predict_fn: Callable[[str], tuple[np.ndarray, np.ndarray]], output_dir: str, plot_fn=plot_prediction, calibration: tuple[float, float] = (1.0, 0.0), collect_holdout: bool = False, collect_train_preds: bool = False) -> dict:
    cal_a, cal_b = calibration
    all_metrics: dict[str, dict] = {}
    holdout_rows: list[dict] = []
    train_true: list[np.ndarray] = []
    train_pred: list[np.ndarray] = []
    raw_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for vid in video_ids:
        split = "val" if vid in val_ids else "train"
        y_true, y_pred = predict_fn(vid)
        raw_preds[vid] = (y_true, y_pred)
        if collect_train_preds and split == "train":
            train_true.append(y_true)
            train_pred.append(y_pred)

        y_pred_cal = (cal_a * y_pred + cal_b).astype(y_pred.dtype)
        metrics = seq_metrics(y_pred_cal, y_true)
        all_metrics[vid] = {**metrics, "split": split, "n_seconds": len(y_true)}
        logger.info("%s [%s]  RMSE=%.4f  MAE=%.4f  r=%.3f", vid, split, metrics["rmse"], metrics["mae"], metrics.get("pearson", metrics.get("spearman", 0)))
        is_ad = video_dfs[vid]["is_ad"].values if "is_ad" in video_dfs[vid].columns else None
        plot_fn(vid, y_true, y_pred_cal, is_ad, split, metrics, os.path.join(output_dir, "videos", vid, "prediction.png"))

        if collect_holdout and split == "val": holdout_rows.extend( {"video": vid, "second": s, "true_retention": y_true[s], "pred_retention": y_pred_cal[s], "abs_error": abs(y_true[s] - y_pred_cal[s])} for s in range(len(y_true)) )

    return {"all_metrics": all_metrics, "holdout_rows": holdout_rows, "train_true": train_true, "train_pred": train_pred, "raw_preds": raw_preds}

def apply_global_calibration(train_true_list: list[np.ndarray], train_pred_list: list[np.ndarray]) -> tuple[float, float]:
    cal_true = np.concatenate(train_true_list)
    cal_pred = np.concatenate(train_pred_list)
    try:
        A = np.vstack([cal_pred, np.ones(len(cal_pred))]).T
        cal_a, _ = np.linalg.lstsq(A, cal_true, rcond=None)[0:2]
        cal_a = float(np.clip(cal_a, 0.5, 2.0))
        cal_b = float(np.mean(cal_true) - cal_a * np.mean(cal_pred))
        cal_b = float(np.clip(cal_b, -50.0, 50.0))
    except Exception:
        cal_a, cal_b = 1.0, 0.0
    logger.info("Global calibration: a=%.4f, b=%.4f", cal_a, cal_b)
    return cal_a, cal_b


def run_tabular_feature_importance_all(args: argparse.Namespace, subdir: str = "tabular_run_all") -> dict[str, Any]:
    out_dir = os.path.join(args.output_dir, "feature_importance", subdir)
    os.makedirs(out_dir, exist_ok=True)
    pipelines = list(PIPELINE_REGISTRY.keys())
    try:
        run_all(output_dir=args.output_dir_features, results_dir=out_dir, target=TABULAR_FEATURE_IMPORTANCE_TARGET, top_n=int(args.feature_importance_top_n), pipelines=pipelines)
        return {
            "tabular_run_all_dir": out_dir,
            "tabular_run_all_target": TABULAR_FEATURE_IMPORTANCE_TARGET,
            "tabular_run_all_top_n": int(args.feature_importance_top_n),
            "tabular_run_all_pipelines": pipelines,
        }
    except Exception:
        logger.exception("Tabular feature_importance.run_all failed")
        return {"tabular_run_all_error": True, "tabular_run_all_dir": out_dir, "tabular_run_all_pipelines": pipelines}


def augmentation_kwargs(args: argparse.Namespace) -> dict[str, float]:
    return {"feature_mask_prob": args.feature_mask_prob if args.use_augmentation else 0.0, "noise_std": args.noise_std if args.use_augmentation else 0.0}


def run_video_clustering_if_requested(args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_video_clustering: return {}
    out_root = str(Path(args.video_clustering_output_dir))
    cmd = [sys.executable, "-m", "analysis.video_clustering", "--data-dir", str(args.snapshot_dir), "--embeddings-dir", str(getattr(args, "embeddings_root", "embeddings")), "--output-dir", str(args.output_dir_features), "--out-root", out_root, "--strategy", str(args.video_clustering_strategy), "--min-k", str(args.video_clustering_min_k), "--max-k", str(args.video_clustering_max_k), "--random-state", str(args.random_seed)]
    try:
        subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2])
        return {"video_clustering_dir": out_root, "video_clustering_strategy": args.video_clustering_strategy}
    except Exception:
        logger.exception("Video clustering failed")
        return {"video_clustering_error": True, "video_clustering_dir": out_root}


def _drop_cli_args(argv: list[str], with_value: set[str], flags: set[str]) -> list[str]:
    out = []
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in flags or key in with_value: continue
        out.append(arg)
    return out


def run_loo_all(args: argparse.Namespace, module_name: str) -> bool:
    if not args.loo_all:
        return False
    ids = sorted(path.name.replace("_features.csv", "") for path in Path(args.output_dir_features).glob("*_features.csv") if not path.name.endswith(".partial"))
    base = Path(args.output_dir or "loo_runs")
    base.mkdir(parents=True, exist_ok=True)
    cli = _drop_cli_args(sys.argv[1:], {"--output-dir", "--eval-video"}, {"--loo-all", "--run-video-clustering"})
    rows = []
    for idx, vid in enumerate(ids):
        out_dir = base / f"loo_{idx:03d}_{vid}"
        subprocess.run([sys.executable, "-m", module_name, *cli, "--eval-video", vid, "--output-dir", str(out_dir)], check=True, cwd=Path(__file__).resolve().parents[2])
        metrics_path = out_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        val_rows = [m for m in payload.get("per_video", {}).values() if m.get("split") == "val"]
        for m in val_rows:
            rows.append({"fold": idx, "video": vid, **{k: v for k, v in m.items() if isinstance(v, (int, float, str))}})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(base / "loo_summary.csv", index=False)
        means = {c: float(df[c].mean()) for c in df.select_dtypes(include=[np.number]).columns if c != "fold"}
        (base / "loo_summary.json").write_text(json.dumps({"n_folds": len(ids), "mean": means}, indent=2), encoding="utf-8")
    return True


def est_importance(model: torch.nn.Module, feature_cols: list[str], video_dfs: dict, val_ids: list[str], video_ids: list[str], normalizer, device: torch.device, args: argparse.Namespace, subdir: str, predict_kwargs: dict | None = None) -> dict[str, Any]:
    ids = val_ids if val_ids else video_ids
    out_dir = os.path.join(args.output_dir, "feature_importance", subdir)
    os.makedirs(out_dir, exist_ok=True)
    meta: dict[str, Any] = {"seq_loss_importance_methods": ["permutation", "median_ablation"], "seq_loss_importance_video_ids": ids, "seq_loss_importance_top_n": int(args.feature_importance_top_n)}
    try:
        compute_predict_video_loss_importance( model, feature_cols, video_dfs, ids, normalizer, device, out_dir, args.window_size, top_n=int(args.feature_importance_top_n), predict_kwargs=predict_kwargs or {}, )
        meta.update({"seq_loss_importance_dir": out_dir})
    except Exception:
        logger.exception("Seq loss feature importance failed")
        meta["seq_loss_importance_error"] = True
        meta["seq_loss_importance_dir"] = out_dir
    meta.update(run_tabular_feature_importance_all(args))
    return meta

run_transformer_feature_importance_suite = est_importance

def save_metrics_json(args: argparse.Namespace, *, model_name: str, feature_cols: list[str], n_feat: int, train_ids: list[str], val_ids: list[str], result: dict, all_metrics: dict, feature_importance_meta: dict | None = None, extra_top_level: dict | None = None, include_config: bool = True) -> None:
    payload = {"model": model_name, "n_features": n_feat, "feature_cols": feature_cols, "train_ids": train_ids, "val_ids": val_ids, "best_val_loss": result.get("best_val_loss", result.get("best_val_mae")), "epochs_trained": result["epochs_trained"], "elapsed_sec": result["elapsed_sec"], "per_video": all_metrics}
    if extra_top_level: payload.update(extra_top_level)
    if feature_importance_meta: payload["feature_importance"] = feature_importance_meta
    if include_config: payload["config"] = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
    Path(os.path.join(args.output_dir, "metrics.json")).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def save_mae_summary(all_metrics: dict, output_dir: str, model_name: str) -> None:
    plot_mae_summary(all_metrics, output_dir, model_name=model_name)

def apply_params(args: argparse.Namespace, model_family: str) -> None:
    if not getattr(args, "tuned_params_json", ""): return
    merge_tuned_file_into_args(args, args.tuned_params_json, model_family=model_family, apply_architecture=args.tuned_apply_architecture, save_copy_to=Path(args.output_dir) / "tuned_params_applied.json")

__all__ = ["add_common_args", "apply_global_calibration", "apply_tabular_pca", "augmentation_kwargs", "build_tuned_feature_filter_kwargs", "compute_baseline_curve", "init_run", "load_and_filter_data", "make_normalizer", "apply_params", "predict_all_videos", "resolve_output_video_ids", "resolve_split", "run_loo_all", "run_tabular_feature_importance_all", "run_transformer_feature_importance_suite", "run_video_clustering_if_requested", "est_importance", "save_mae_summary", "save_metrics_json", "set_model_baseline"]