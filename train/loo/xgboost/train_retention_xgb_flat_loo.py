from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from train.common.retention_data_layer import DEFAULT_PARENT_FOLDER_ID
from train.loo.common import LooArtifacts, clip01, curve_metrics, load_loo_data, print_run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LOO-модель: XGBoost в flattened-формате. Вместо 50 отдельных моделей "
            "по 89 строк — одна модель на 89×50=4450 строк. "
            "Фичи = video_features + point_position. "
            "Модель обучается на ВСЕХ точках сразу и видит межточечные паттерны."
        )
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--snapshot-dir", default="")
    parser.add_argument("--root-folder-id", default=DEFAULT_PARENT_FOLDER_ID)
    parser.add_argument("--limit-videos", type=int, default=90)
    parser.add_argument("--train-videos", type=int, default=89)
    parser.add_argument("--curve-points", type=int, default=50)
    parser.add_argument("--eval-video-folder", default="")
    parser.add_argument("--eval-drive-file-id", default="")
    parser.add_argument("--output-dir", default="xgb_flat_experiment")

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--reg-alpha", type=float, default=0.5)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def _flatten_to_rows(x_video: pd.DataFrame, y_matrix: np.ndarray, curve_points: int) -> tuple[pd.DataFrame, np.ndarray]:
    n_videos = len(x_video)
    rows = []
    targets = []
    for vid_idx in range(n_videos):
        base = x_video.iloc[vid_idx].to_dict()
        for p in range(curve_points):
            row = dict(base)
            row["__point_idx__"] = float(p)
            row["__point_norm__"] = float(p / max(1, curve_points - 1))
            row["__point_norm_sq__"] = row["__point_norm__"] ** 2
            rows.append(row)
            targets.append(float(y_matrix[vid_idx, p]))
    return pd.DataFrame(rows).fillna(0.0), np.array(targets, dtype=float)


def _flatten_test(x_video: pd.DataFrame, curve_points: int) -> pd.DataFrame:
    base = x_video.iloc[0].to_dict()
    rows = []
    for p in range(curve_points):
        row = dict(base)
        row["__point_idx__"] = float(p)
        row["__point_norm__"] = float(p / max(1, curve_points - 1))
        row["__point_norm_sq__"] = row["__point_norm__"] ** 2
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    data = load_loo_data(args, empty_label="xgb_flat")
    X_train_video, X_test_video, y_train_mat, y_true = data.x_train, data.x_test, data.y_train, data.y_true
    n_points = int(args.curve_points)
    n_train = len(X_train_video)

    X_train_flat, y_train_flat = _flatten_to_rows(X_train_video, y_train_mat, n_points)
    X_test_flat = _flatten_test(X_test_video, n_points)

    print(f"[xgb_flat] train_videos={n_train} points={n_points} flat_rows={len(X_train_flat)} features={X_train_flat.shape[1]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        random_state=args.random_seed,
        verbosity=0,
    )

    print("[xgb_flat] stage=fit")
    model.fit(X_train_flat, y_train_flat)

    print("[xgb_flat] stage=predict")
    y_pred_raw = model.predict(X_test_flat)
    y_pred = clip01(np.array(y_pred_raw, dtype=float))

    model.save_model(str(out_dir / "xgb_flat_model.json"))

    abs_err = np.abs(y_pred - y_true)
    cm = curve_metrics(y_pred, y_true)

    result_df = pd.DataFrame(
        {
            "point_idx": list(range(n_points)),
            "point_frac": np.linspace(0.0, 1.0, n_points),
            "pred_retention": y_pred,
            "pred_retention_norm": y_pred,
            "pred_score_raw": y_pred,
            "true_retention": y_true,
            "abs_error": abs_err,
        }
    )

    artifacts = LooArtifacts(out_dir, "xgb_flat_dataset.csv")
    artifacts.write_tables(data.all_df, result_df)

    metrics: dict[str, Any] = {
        "videos_total_with_target": len(data.rows),
        "videos_used": len(data.all_df),
        "train_videos": n_train,
        "curve_points": n_points,
        "flat_train_rows": len(X_train_flat),
        "n_features": int(X_train_flat.shape[1]),
        "test_video": str(data.test_df.iloc[0]["video_folder"]),
        "test_drive_file_id": str(data.test_df.iloc[0]["drive_file_id"]),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "dataset_path": str(artifacts.dataset_path),
        "prediction_path": str(artifacts.prediction_path),
        **cm,
    }
    artifacts.write_metrics(metrics)
    print_run_report("Retention XGBoost-Flat LOO", metrics, result_df)
    return metrics


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
