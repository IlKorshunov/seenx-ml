"""
Modes:
    simple    — Ad-aware MSE loss with manual loop.
    composite — Composite loss + SWA + baseline + tuned-params.

python train_transformer_seq_v1.py --mode simple --eval-video DhFuAhFMvms
python train_transformer_seq_v1.py --mode composite --top-k-features 40 --epochs 300
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.retention_transformer import RetentionTransformer
from train.common.pipeline import *
from train.common.retention_plots import plot_prediction, plot_retention_prediction, plot_training_curve
from train.common.seq_data_utils import WindowedSeqDataset, ad_aware_loss, predict_video, time_feature_extra_dim
from train.lstm.lstm_seq_base import run_sequence_training_loop
from train.transformer.transformer_base import run_transformer_feature_importance_suite


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RetentionTransformer.")
    add_common_args(p)
    p.add_argument("--mode", choices=["simple", "composite"], default="composite")
    p.add_argument("--head-type", choices=["cumulative", "sigmoid", "tanh"], default="tanh")
    p.add_argument("--no-baseline", action="store_true", default=False)
    p.add_argument("--emb-pca-components", type=int, default=36)
    return p.parse_args()


def _train_simple(model, train_dl, val_dl, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    pointwise_loss_fn = nn.MSELoss(reduction="none")
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))

    best_val, no_improve, best_state = float("inf"), 0, {}
    train_losses, val_losses = [], []
    t0, epoch = time.time(), 0

    def _to_dev(batch, *keys): return tuple(batch[k].to(device) for k in keys)
    for epoch in (bar := tqdm(range(1, args.epochs + 1), desc="Training", unit="ep")):
        model.train()
        loss_sum, n_pts = 0.0, 0
        for batch in tqdm(train_dl, desc=f"Ep {epoch} [train]", leave=False, unit="b"):
            feats, tgt, pad, ad = _to_dev(batch, "features", "retention", "padding_mask", "is_ad")
            engagement_weights = batch["video_weight"].to(device) if args.engagement_weight else None
            loss = ad_aware_loss(model(feats, src_key_padding_mask=pad), tgt, ad, pad, pointwise_loss_fn, args.ad_penalty_weight, video_weight=engagement_weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n = (~pad).sum().item()
            loss_sum += loss.item() * n
            n_pts += n
        scheduler.step()
        train_losses.append(loss_sum / max(n_pts, 1))

        model.eval()
        loss_sum, n_pts = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Ep {epoch} [val]", leave=False, unit="b"):
                feats, tgt, pad, ad = _to_dev(batch, "features", "retention", "padding_mask", "is_ad")
                loss = ad_aware_loss(model(feats, src_key_padding_mask=pad), tgt, ad, pad, pointwise_loss_fn, 1.0)
                n = (~pad).sum().item()
                loss_sum += loss.item() * n
                n_pts += n
        val_losses.append(loss_sum / max(n_pts, 1))

        bar.set_postfix(train=f"{train_losses[-1]:.4f}", val=f"{val_losses[-1]:.4f}")
        writer.add_scalar("MAE/train", train_losses[-1], epoch)
        writer.add_scalar("MAE/val", val_losses[-1], epoch)

        if val_losses[-1] < best_val:
            best_val, no_improve = val_losses[-1], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stop at epoch %d", epoch)
                break

    writer.close()
    model.load_state_dict(best_state)
    return model, {"train_losses": train_losses, "val_losses": val_losses, "best_val_loss": round(best_val, 6), "epochs_trained": epoch, "elapsed_sec": round(time.time() - t0, 1)}


def _train_composite(model, train_dl, val_dl, device, args):
    model, result = run_sequence_training_loop(model, train_dl, val_dl, device, args, use_engagement_weight=args.engagement_weight)
    return model.to(device), result


def _write_holdout_rows(pred_out: dict, output_dir: str) -> None:
    if pred_out["holdout_rows"]:
        pd.DataFrame(pred_out["holdout_rows"]).to_csv(os.path.join(output_dir, "holdout_prediction_vs_true.csv"), index=False)


def main():
    args = parse_args()
    composite = args.mode == "composite"
    if not args.output_dir:
        args.output_dir = "transformer_exp"
    if run_loo_all(args, "train.transformer.train_transformer_seq_v1"):
        return
    device = init_run(args)

    if composite:
        apply_params(args, "tabular_transformer")
        args.alpha_mono = 0.0 if args.head_type in ("cumulative", "sigmoid") else args.alpha_mono
        mode_cfg = {
            "extra_load": {"embeddings_root": "embeddings", "emb_pca_components": args.emb_pca_components, "min_duration_sec": args.min_duration_sec, "max_duration_sec": args.max_duration_sec},
            "ref_sec": lambda train_ref_sec: train_ref_sec,
            "train_dataset_kwargs": lambda train_ref_sec, weights: {"video_weights": weights, **augmentation_kwargs(args), "time_feature_mode": args.time_features, "ref_time_sec_max": train_ref_sec},
            "val_dataset_kwargs": lambda train_ref_sec: {"time_feature_mode": args.time_features, "ref_time_sec_max": train_ref_sec},
            "time_dim": time_feature_extra_dim(args.time_features),
            "model_extra": {"head_type": args.head_type},
            "set_baseline": (lambda model, dfs, ids, norm: set_model_baseline(model, dfs, ids, norm)) if not args.no_baseline and args.head_type == "tanh" else (lambda *unused: None),
            "train_fn": _train_composite,
            "plot_title": "Transformer",
            "pred_kw": lambda train_ref_sec: {"time_feature_mode": args.time_features, "ref_time_sec_max": train_ref_sec},
            "plot_fn": plot_retention_prediction,
            "collect_holdout": False,
            "save_summary": lambda metrics: save_mae_summary(metrics, args.output_dir, "Transformer"),
            "metrics_extra": {"head_type": args.head_type},
            "ckpt_extra": lambda norm: {"head_type": args.head_type},
            "write_holdout": lambda pred_out: None,
        }
    else:
        mode_cfg = {
            "extra_load": {},
            "ref_sec": lambda train_ref_sec: 1.0,
            "train_dataset_kwargs": lambda train_ref_sec, weights: {"video_weights": weights, **augmentation_kwargs(args)},
            "val_dataset_kwargs": lambda train_ref_sec: {},
            "time_dim": 0,
            "model_extra": {},
            "set_baseline": lambda *unused: None,
            "train_fn": _train_simple,
            "plot_title": "",
            "pred_kw": lambda train_ref_sec: {"apply_smoothing": args.apply_smoothing},
            "plot_fn": plot_prediction,
            "collect_holdout": True,
            "save_summary": lambda metrics: None,
            "metrics_extra": {},
            "ckpt_extra": lambda norm: {"normalizer_mean": norm.median.tolist(), "normalizer_std": norm.iqr.tolist()},
            "write_holdout": lambda pred_out: _write_holdout_rows(pred_out, args.output_dir),
        }

    video_dfs, video_ids, output_video_ids, feature_cols = load_and_filter_data(args, extra_load_kwargs=mode_cfg["extra_load"])
    train_ids, val_ids = resolve_split(args, video_ids, output_video_ids)
    video_dfs, feature_cols = apply_tabular_pca(args, video_dfs, train_ids, feature_cols)
    normalizer, ref_sec, video_weights = make_normalizer(args, video_dfs, train_ids, feature_cols)
    ref_sec = mode_cfg["ref_sec"](ref_sec)

    train_dataset_kwargs: dict[str, Any] = mode_cfg["train_dataset_kwargs"](ref_sec, video_weights)
    val_dataset_kwargs: dict[str, Any] = mode_cfg["val_dataset_kwargs"](ref_sec)

    train_ds = WindowedSeqDataset(video_dfs, train_ids, feature_cols, normalizer, args.window_size, args.window_stride, **train_dataset_kwargs)
    val_ds = WindowedSeqDataset(video_dfs, val_ids, feature_cols, normalizer, args.window_size, args.window_stride, **val_dataset_kwargs)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=composite and len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info("Train windows: %d, Val windows: %d", len(train_ds), len(val_ds))

    n_feat = len(feature_cols) + mode_cfg["time_dim"]
    model_kw: dict[str, Any] = dict(n_features=n_feat, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout)
    model_kw.update(mode_cfg["model_extra"])
    model = RetentionTransformer(**model_kw).to(device)
    mode_cfg["set_baseline"](model, video_dfs, train_ids, normalizer)
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    plot_curve_path = os.path.join(args.output_dir, "training_curve.png")
    model, result = mode_cfg["train_fn"](model, train_dl, val_dl, device, args)
    plot_training_curve(result["train_losses"], result["val_losses"], plot_curve_path, mode_cfg["plot_title"])
    pred_kw = mode_cfg["pred_kw"](ref_sec)
    pred_out = predict_all_videos( video_ids=video_ids, val_ids=val_ids, video_dfs=video_dfs, predict_fn=lambda vid: predict_video(model, video_dfs[vid], feature_cols, normalizer, device, args.window_size, **pred_kw), output_dir=args.output_dir, plot_fn=mode_cfg["plot_fn"], collect_holdout=mode_cfg["collect_holdout"], )
    all_metrics = pred_out["all_metrics"]
    mode_cfg["save_summary"](all_metrics)

    fi_meta = run_transformer_feature_importance_suite(model, feature_cols, video_dfs, val_ids, video_ids, normalizer, device, args, "seq_retention_transformer_loss", predict_kwargs=pred_kw)
    fi_meta.update(run_video_clustering_if_requested(args))

    result["best_val_loss"] = result.get("best_val_loss", result.get("best_val_mae"))
    extra = {"mode": args.mode, **mode_cfg["metrics_extra"]}
    save_metrics_json(args, model_name="RetentionTransformer", feature_cols=feature_cols, n_feat=n_feat, train_ids=train_ids, val_ids=val_ids, result=result, all_metrics=all_metrics, feature_importance_meta=fi_meta, extra_top_level=extra, include_config=composite)

    ckpt = {"model_state_dict": model.state_dict(), "n_features": n_feat, "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers, "d_ff": args.d_ff, "dropout": args.dropout, "feature_cols": feature_cols, "normalizer_median": normalizer.median.tolist(), "normalizer_iqr": normalizer.iqr.tolist(), "ret_min": normalizer.ret_min, "ret_max": normalizer.ret_max}
    ckpt.update(mode_cfg["ckpt_extra"](normalizer))

    torch.save(ckpt, os.path.join(args.output_dir, "transformer_model.pt"))
    mode_cfg["write_holdout"](pred_out)
    logger.info("Done (mode=%s). Best val=%.4f, epochs=%d, time=%.0fs", args.mode, result["best_val_loss"], result["epochs_trained"], result["elapsed_sec"])

if __name__ == "__main__":
    main()
