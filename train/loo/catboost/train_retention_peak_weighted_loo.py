from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from tqdm import tqdm

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID
from train.loo.common import LooArtifacts, clip01, curve_metrics, load_loo_data, print_run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("LOO-модель: pointwise regressor с усиленным весом пиков/просадок в train-лоссе, без явного postprocess-занижения рекламы."))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--snapshot-dir", default="drive_snapshot_90")
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    parser.add_argument("--limit-videos", type=int, default=45)
    parser.add_argument("--train-videos", type=int, default=44)
    parser.add_argument("--curve-points", type=int, default=50)
    parser.add_argument("--eval-video-folder", default="")
    parser.add_argument("--eval-drive-file-id", default="")
    parser.add_argument("--output-dir", default="peak_weighted_experiment")

    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--task-type", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--weight-slope-alpha", type=float, default=1.8)
    parser.add_argument("--weight-curvature-alpha", type=float, default=1.2)
    parser.add_argument("--weight-max", type=float, default=7.0)
    return parser.parse_args()


def _point_weights(y_train: np.ndarray, point_idx: int, slope_alpha: float, curvature_alpha: float, w_max: float) -> np.ndarray:
    n, _ = y_train.shape
    if n == 0:
        return np.zeros((0,), dtype=float)

    prev_idx = max(0, point_idx - 1)
    prev2_idx = max(0, point_idx - 2)
    slope = np.abs(y_train[:, point_idx] - y_train[:, prev_idx])
    curvature = np.abs(y_train[:, point_idx] - 2.0 * y_train[:, prev_idx] + y_train[:, prev2_idx])

    slope_scale = float(np.median(slope[slope > 1e-9])) if np.any(slope > 1e-9) else 1.0
    curv_scale = float(np.median(curvature[curvature > 1e-9])) if np.any(curvature > 1e-9) else 1.0

    slope_norm = slope / max(1e-6, slope_scale)
    curv_norm = curvature / max(1e-6, curv_scale)
    w = 1.0 + float(slope_alpha) * slope_norm + float(curvature_alpha) * curv_norm
    return np.clip(w, 0.2, max(1.0, float(w_max)))


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    data = load_loo_data(args, empty_label="peak_weighted")
    x_train, x_test, y_train, y_true = data.x_train, data.x_test, data.y_train, data.y_true

    y_pred = np.zeros((args.curve_points,), dtype=float)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "point_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(range(args.curve_points), desc="Training points"):
        target = y_train[:, p]
        w = _point_weights(
            y_train=y_train, point_idx=p, slope_alpha=float(args.weight_slope_alpha), curvature_alpha=float(args.weight_curvature_alpha), w_max=float(args.weight_max)
        )
        if target.size == 0:
            y_pred[p] = 0.0
            continue
        if float(np.var(target)) < 1e-12:
            y_pred[p] = float(target[0])
            continue
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            random_seed=args.random_seed + p,
            task_type=getattr(args, "task_type", "GPU"),
            gpu_ram_part=getattr(args, "gpu_ram_part", 0.6),
            verbose=False,
        )
        model.fit(x_train, target, sample_weight=w)
        y_pred[p] = float(model.predict(x_test)[0])
        model.save_model(str(models_dir / f"catboost_point_{p:03d}.cbm"))

    y_pred = clip01(y_pred)
    abs_err = np.abs(y_pred - y_true)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(args.curve_points)),
            "point_frac": np.linspace(0.0, 1.0, args.curve_points),
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    dataset_path = out_dir / "peak_weighted_dataset.csv"
    pred_path = out_dir / "holdout_prediction_vs_true.csv"
    artifacts = LooArtifacts(out_dir, "peak_weighted_dataset.csv")
    artifacts.write_tables(data.all_df, result_df)

    cm = curve_metrics(y_pred, y_true)
    metrics = {
        "videos_total_with_target": len(data.rows),
        "videos_used": len(data.all_df),
        "train_videos": len(data.train_df),
        "curve_points": int(args.curve_points),
        "test_video": str(data.test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(data.test_df.iloc[0]["drive_file_id"]),
        "weight_slope_alpha": float(args.weight_slope_alpha),
        "weight_curvature_alpha": float(args.weight_curvature_alpha),
        "weight_max": float(args.weight_max),
        "dataset_path": str(artifacts.dataset_path),
        "prediction_path": str(artifacts.prediction_path),
        "models_dir": str(models_dir),
        **cm,
    }
    artifacts.write_metrics(metrics)
    print_run_report("Retention Peak-Weighted LOO", metrics)
    return metrics


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
