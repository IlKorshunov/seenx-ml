from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID
from train.loo.common import LooArtifacts, clip01, curve_metrics, load_loo_data, print_run_report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--snapshot-dir", default="")
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    parser.add_argument("--limit-videos", type=int, default=90)
    parser.add_argument("--train-videos", type=int, default=89)
    parser.add_argument("--curve-points", type=int, default=50)
    parser.add_argument("--eval-video-folder", default="")
    parser.add_argument("--eval-drive-file-id", default="")
    parser.add_argument("--output-dir", default="blended_quantile_experiment")

    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--gpu-ram-part", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--baseline-weight", type=float, default=0.35, help="Вес baseline в финальном прогнозе [0..1]. Остальное — модель.")
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    data = load_loo_data(args, empty_label="blended_quantile", reset_index=False)
    X_train, X_test, y_train, y_true = data.x_train, data.x_test, data.y_train, data.y_true
    total_points = int(args.curve_points)
    baseline_curve = clip01(np.mean(y_train, axis=0))
    alpha = float(np.clip(args.baseline_weight, 0.0, 1.0))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "point_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_pred = np.zeros((total_points,), dtype=float)
    model_count = 0

    for p in range(total_points):
        print(f"[blended_quantile][point] {p + 1}/{total_points}")
        target = y_train[:, p]

        if target.size == 0:
            model_pred[p] = baseline_curve[p]
        elif float(np.var(target)) < 1e-12:
            model_pred[p] = float(target[0])
        else:
            model = CatBoostRegressor(
                loss_function="Quantile:alpha=0.5",
                eval_metric="RMSE",
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
                random_seed=args.random_seed + p,
                task_type="GPU",
                gpu_ram_part=getattr(args, "gpu_ram_part", 0.6),
                verbose=False,
            )
            model.fit(X_train, target)
            model_pred[p] = float(model.predict(X_test)[0])
            model.save_model(str(models_dir / f"quantile_point_{p:03d}.cbm"))
            model_count += 1

    model_pred = clip01(model_pred)
    y_pred = clip01(alpha * baseline_curve + (1.0 - alpha) * model_pred)
    abs_err = np.abs(y_pred - y_true)
    cm = curve_metrics(y_pred, y_true)
    result_df = pd.DataFrame(
        {
            "point_idx": list(range(total_points)),
            "baseline_value": baseline_curve,
            "model_quantile_pred": model_pred,
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    artifacts = LooArtifacts(out_dir, "blended_quantile_dataset.csv")
    artifacts.write_tables(data.all_df, result_df)

    metrics: dict[str, Any] = {
        "videos_total_with_target": len(data.rows),
        "videos_used": len(data.all_df),
        "train_videos": len(data.train_df),
        "curve_points": total_points,
        "test_video": str(data.test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(data.test_df.iloc[0]["drive_file_id"]),
        "baseline_weight": alpha,
        "model_count": model_count,
        "dataset_path": str(artifacts.dataset_path),
        "prediction_path": str(artifacts.prediction_path),
        "models_dir": str(models_dir),
        **cm,
    }
    artifacts.write_metrics(metrics)
    print_run_report("Retention Blended-Quantile LOO", metrics, result_df)
    return metrics

if __name__ == "__main__":
    run_experiment(parse_args())
