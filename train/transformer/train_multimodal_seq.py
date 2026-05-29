"""
Multimodal retention trainer: embeddings (visual 768 + audio 512 + text 256) + tabular.
python train_multimodal_seq.py --arch transformer --val-first-n-output 10 --epochs 200
python train_multimodal_seq.py --arch lstm --val-first-n-output 10 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.retention_multimodal_lstm import MultimodalRetentionLSTM
from src.models.retention_multimodal_transformer import MultimodalRetentionTransformer
from src.utils.embedding_aligner import AUDIO_DIM, TEXT_DIM, VISUAL_DIM
from train.common.composite_trainer import run_composite_training_loop, to_device_batch
from train.common.pipeline import *
from train.common.retention_plots import COLOR_ACTUAL as C_BLUE, GRID_ALPHA, plot_prediction, plot_training_curve, save_figure as _save_fig
from train.common.seq_data_utils import *
from train.common.tuned_params_io import apply_best_params_to_args
from train.transformer.transformer_base import run_tabular_feature_importance_all


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multimodal retention model.")
    add_common_args(p)
    p.add_argument("--arch", default="transformer", choices=["transformer", "lstm"])
    p.add_argument("--embeddings-root", default="embeddings")
    p.add_argument("--use-conv-blocks", action="store_true", default=False)
    p.add_argument("--global-calibration", action="store_true")
    p.add_argument("--tune-first", action="store_true")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--epochs-per-trial", type=int, default=80)
    p.add_argument("--tune-output-dir", default="src/tune_hp/results")
    p.add_argument("--study-name", default="")
    return p.parse_args()


def _run_optuna_and_apply(args: argparse.Namespace) -> argparse.Namespace:
    tune_arch = f"multimodal_{args.arch}"
    study_name = args.study_name or f"tune_{tune_arch}"
    tune_script = Path(__file__).resolve().parents[2] / "src" / "tune_hp" / "tune.py"
    cmd = [sys.executable, str(tune_script), "--arch", tune_arch, "--output-dir-features", args.output_dir_features, "--snapshot-dir", args.snapshot_dir, "--embeddings-root", args.embeddings_root, "--val-first-n-output", str(args.val_first_n_output), "--n-trials", str(args.n_trials), "--epochs-per-trial", str(args.epochs_per_trial), "--device", args.device, "--output-dir", args.tune_output_dir, "--study-name", study_name, "--random-seed", str(args.random_seed)]
    cmd.append("--use-curve-raw" if args.use_curve_raw else "--no-use-curve-raw")
    logger.info("Running Optuna tuning: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    best = json.loads((Path(args.tune_output_dir) / f"{study_name}_best.json").read_text(encoding="utf-8"))
    apply_best_params_to_args(args, best.get("best_params", {}), model_family=f"multimodal_{args.arch}", apply_architecture=True)
    out = Path(args.output_dir) / "tuned_params_applied.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return args


def _multimodal_preds(model, video_dfs, video_embeddings, ids, feature_cols, normalizer, device, args, ref_sec, cal_a, cal_b) -> dict[str, np.ndarray]:
    preds = {}
    for vid in ids:
        _, y_pred = predict_video_multimodal(
            model, video_dfs[vid], video_embeddings.get(vid), feature_cols, normalizer, device,
            args.window_size, smooth_window=0, apply_smoothing=False,
            time_feature_mode=args.time_features, ref_time_sec_max=ref_sec,
        )
        preds[vid] = (cal_a * y_pred + cal_b).astype(y_pred.dtype)
    return preds


def _median_fill(fn: str, video_dfs: dict[str, pd.DataFrame], ids: list[str]) -> float:
    parts = [video_dfs[v][fn].values for v in ids if fn in video_dfs[v].columns]
    return float(np.nanmedian(np.concatenate(parts))) if parts else 0.0


def _mean_abs_delta(base, ablated, ids):
    return float(np.mean([float(np.mean(np.abs(base[v] - ablated[v]))) for v in ids]))


def _mean_mae(preds, y_true_by_vid, ids):
    return float(np.mean([float(np.mean(np.abs(preds[v] - y_true_by_vid[v]))) for v in ids]))


def _tabular_pred_delta(baseline_preds, feature_cols, video_dfs, video_embeddings, ids, normalizer, model, device, args, ref_sec, cal_a, cal_b) -> pd.DataFrame:
    rows = []
    for fi, fn in enumerate(feature_cols):
        fill = _median_fill(fn, video_dfs, ids)
        perturbed = {vid: video_dfs[vid].assign(**({fn: fill} if fn in video_dfs[vid].columns else {})) for vid in ids}
        ablated = _multimodal_preds(model, perturbed, video_embeddings, ids, feature_cols, normalizer, device, args, ref_sec, cal_a, cal_b)
        rows.append({"feature": fn, "feature_idx": fi, "importance_pred_delta": _mean_abs_delta(baseline_preds, ablated, ids), "fill_value": fill})
    return pd.DataFrame(rows).sort_values("importance_pred_delta", ascending=False).reset_index(drop=True)


def _modality_ablation(baseline_preds, y_true_by_vid, baseline_mae, model, video_dfs, video_embeddings, ids, feature_cols, normalizer, device, args, ref_sec, cal_a, cal_b) -> pd.DataFrame:
    slices = [("visual", 0, VISUAL_DIM), ("audio", VISUAL_DIM, VISUAL_DIM + AUDIO_DIM), ("text", VISUAL_DIM + AUDIO_DIM, VISUAL_DIM + AUDIO_DIM + TEXT_DIM)]
    rows = {}
    for name, s, e in slices:
        perturbed = {}
        for vid in ids:
            emb_v = video_embeddings.get(vid)
            if emb_v is not None:
                emb_ablated = emb_v.copy()
                emb_ablated[:, s:e] = 0.0
            else:
                emb_ablated = None
            perturbed[vid] = emb_ablated
        mod_preds = _multimodal_preds(model, video_dfs, perturbed, ids, feature_cols, normalizer, device, args, ref_sec, cal_a, cal_b)
        rows[name] = {
            "pred_delta_mean_abs": round(_mean_abs_delta(baseline_preds, mod_preds, ids), 6),
            "mae_without": round(_mean_mae(mod_preds, y_true_by_vid, ids), 4),
            "mae_increase": round(_mean_mae(mod_preds, y_true_by_vid, ids) - baseline_mae, 4),
        }
        logger.info("Ablation %s: pred_delta=%.6f, mae_increase=+%.4f", name, rows[name]["pred_delta_mean_abs"], rows[name]["mae_increase"])

    tab_only = _multimodal_preds(model, video_dfs, {vid: None for vid in ids}, ids, feature_cols, normalizer, device, args, ref_sec, cal_a, cal_b)
    rows["all_embeddings_zeroed"] = {
        "pred_delta_mean_abs": round(_mean_abs_delta(baseline_preds, tab_only, ids), 6),
        "mae_without": round(_mean_mae(tab_only, y_true_by_vid, ids), 4),
        "mae_increase": round(_mean_mae(tab_only, y_true_by_vid, ids) - baseline_mae, 4),
    }
    return pd.DataFrame([{"modality": k, **v} for k, v in rows.items()])


def _run_feature_importance(model, feature_cols, video_dfs, video_embeddings, val_ids, video_ids, normalizer, device, args, ref_sec, cal_a, cal_b) -> dict:
    ids = val_ids if val_ids else video_ids
    out_dir = Path(args.output_dir) / "feature_importance" / f"multimodal_{args.arch}_loss"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    y_true_by_vid = {vid: pd.to_numeric(video_dfs[vid]["retention"], errors="coerce").fillna(0).values.astype(np.float64) for vid in ids}
    baseline_preds = _multimodal_preds(model, video_dfs, video_embeddings, ids, feature_cols, normalizer, device, args, ref_sec, cal_a, cal_b)
    baseline_mae = _mean_mae(baseline_preds, y_true_by_vid, ids)

    pred_df = _tabular_pred_delta(baseline_preds, feature_cols, video_dfs, video_embeddings, ids, normalizer, model, device, args, ref_sec, cal_a, cal_b)
    pred_df.to_csv(out_dir / "tabular_pred_delta_importance.csv", index=False)

    mod_df = _modality_ablation(baseline_preds, y_true_by_vid, baseline_mae, model, video_dfs, video_embeddings, ids, feature_cols, normalizer, device, args, ref_sec, cal_a, cal_b)
    mod_df.to_csv(out_dir / "modality_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(mod_df["modality"][::-1], mod_df["pred_delta_mean_abs"][::-1], color=C_BLUE)
    ax.axvline(0, color="black", lw=0.5)
    ax.set(xlabel="Mean |pred_base − pred_ablated|", title="Modality ablation")
    ax.grid(True, alpha=GRID_ALPHA, axis="x")
    plt.tight_layout()
    _save_fig(fig, str(out_dir / "modality_ablation.png"))
    return {
        "multimodal_loss_importance_dir": str(out_dir),
        "multimodal_loss_importance_baseline_mae": baseline_mae,
        "multimodal_loss_importance_methods": ["tabular_median_ablation_pred_delta", "modality_ablation"],
        "multimodal_loss_importance_video_ids": ids,
    }


def _train(model, train_dl, val_dl, device, args, use_engagement_weight):
    def _forward(m, batch, dev):
        emb, tab, tgt, pad, ad, spikes = to_device_batch(batch, dev, "embeddings", "tabular", "retention", "padding_mask", "is_ad", "spike_triggers")
        engagement_weights = batch["video_weight"].to(dev) if use_engagement_weight else None
        return m(emb, tabular=tab, src_key_padding_mask=pad), tgt, ad, spikes, pad, engagement_weights
    return run_composite_training_loop(model, train_dl, val_dl, device, args, _forward, enable_swa=True)

def main():
    args = parse_args()
    if not args.output_dir:
        args.output_dir = os.path.join("transformer_exp" if args.arch == "transformer" else "lstm_exp", "multimodal_latest")
    if run_loo_all(args, "train.transformer.train_multimodal_seq"):
        return
    device = init_run(args)

    if args.tune_first:
        args = _run_optuna_and_apply(args)
    elif args.tuned_params_json:
        apply_params(args, f"multimodal_{args.arch}")

    extra_load = {
        "emb_pca_components": 0,
        "min_duration_sec": args.min_duration_sec,
        "max_duration_sec": args.max_duration_sec,
    }
    video_dfs, video_ids, output_video_ids, feature_cols = load_and_filter_data(args, extra_load_kwargs=extra_load)

    logger.info("Loading aligned embeddings")
    video_embeddings = load_aligned_embeddings_for_videos(video_dfs, args.embeddings_root)
    video_dfs = resample_video_dfs_to_curve_points(video_dfs, args.curve_points)
    video_embeddings = resample_embeddings_to_match_dfs(video_embeddings, video_dfs)

    train_ids, val_ids = resolve_split(args, video_ids, output_video_ids)
    video_dfs, feature_cols = apply_tabular_pca(args, video_dfs, train_ids, feature_cols)
    normalizer, ref_sec, video_weights = make_normalizer(args, video_dfs, train_ids, feature_cols)

    common_ds_kw = {"time_feature_mode": args.time_features, "ref_time_sec_max": ref_sec}
    train_ds = MultimodalWindowedDataset(video_dfs, video_embeddings, train_ids, feature_cols, normalizer, args.window_size, args.window_stride, video_weights=video_weights, **augmentation_kwargs(args), **common_ds_kw)
    val_ds = MultimodalWindowedDataset(video_dfs, video_embeddings, val_ids, feature_cols, normalizer, args.window_size, args.window_stride, **common_ds_kw)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info("Train windows: %d, Val windows: %d", len(train_ds), len(val_ds))

    n_tab = len(feature_cols) + time_feature_extra_dim(args.time_features)
    if args.arch == "lstm":
        model = MultimodalRetentionLSTM(hidden_size=args.d_model, n_layers=args.n_layers, dropout=args.dropout, n_tabular_features=n_tab, use_conv_blocks=args.use_conv_blocks).to(device)
    else:
        model = MultimodalRetentionTransformer(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout, n_tabular_features=n_tab, use_conv_blocks=args.use_conv_blocks).to(device)

    set_model_baseline(model, video_dfs, train_ids, normalizer)
    logger.info("Model %s: %d params, %d tab features", args.arch, sum(p.numel() for p in model.parameters()), n_tab)

    model, result = _train(model, train_dl, val_dl, device, args, args.engagement_weight)
    model = model.to(device)
    plot_training_curve(result["train_losses"], result["val_losses"], os.path.join(args.output_dir, "training_curve.png"))

    def _predict_fn_raw(vid):
        return predict_video_multimodal(model, video_dfs[vid], video_embeddings.get(vid), feature_cols, normalizer, device, args.window_size, smooth_window=args.smooth_window, apply_smoothing=args.apply_smoothing, time_feature_mode=args.time_features, ref_time_sec_max=ref_sec)

    if args.global_calibration:
        train_true, train_pred = [], []
        for vid in train_ids:
            yt, yp = _predict_fn_raw(vid)
            train_true.append(yt)
            train_pred.append(yp)
        cal_a, cal_b = apply_global_calibration(train_true, train_pred)
    else:
        cal_a, cal_b = 1.0, 0.0
        logger.info("Without calibration")

    pred_out = predict_all_videos(video_ids=video_ids, val_ids=val_ids, video_dfs=video_dfs, predict_fn=_predict_fn_raw, output_dir=args.output_dir, plot_fn=plot_prediction, calibration=(cal_a, cal_b))
    all_metrics = pred_out["all_metrics"]

    model_label = "Multimodal-Transformer" if args.arch == "transformer" else "Multimodal-LSTM"
    save_mae_summary(all_metrics, args.output_dir, model_label)

    logger.info("Computing feature importance")
    try:
        fi_meta = _run_feature_importance(model, feature_cols, video_dfs, video_embeddings, val_ids, video_ids, normalizer, device, args, ref_sec, cal_a, cal_b)
    except Exception:
        logger.exception("Multimodal feature importance failed")
        fi_meta = {"multimodal_loss_importance_error": True}
    fi_meta.update(run_tabular_feature_importance_all(args))
    fi_meta.update(run_video_clustering_if_requested(args))

    save_metrics_json( args, model_name=f"Multimodal{args.arch.title()}", feature_cols=feature_cols, n_feat=n_tab, train_ids=train_ids, val_ids=val_ids, result=result, all_metrics=all_metrics, feature_importance_meta=fi_meta, extra_top_level={"arch": args.arch, "n_tabular_features": n_tab}, )

    torch.save({
        "model_state_dict": model.state_dict(), "arch": args.arch, "n_tabular_features": n_tab,
        "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers,
        "d_ff": args.d_ff, "dropout": args.dropout, "feature_cols": feature_cols,
        "normalizer_median": normalizer.median.tolist(), "normalizer_iqr": normalizer.iqr.tolist(),
        "ret_min": normalizer.ret_min, "ret_max": normalizer.ret_max,
    }, os.path.join(args.output_dir, f"multimodal_{args.arch}_model.pt"))

    logger.info("Done. Best val=%.4f, epochs=%d, time=%.0fs", result["best_val_loss"], result["epochs_trained"], result["elapsed_sec"])


if __name__ == "__main__":
    main()