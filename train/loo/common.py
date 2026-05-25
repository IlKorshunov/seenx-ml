from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import numpy as np
import pandas as pd

from train.common.retention_data_layer import _point_col, _safe_float, build_rows_with_targets_source, make_feature_matrix, select_train_test

from src.normalize.curves import clip_unit_interval as clip01


def target_matrix(df: pd.DataFrame, curve_points: int) -> np.ndarray:
    return clip01(np.array([[_safe_float(row[_point_col(p)], 0.0) for p in range(curve_points)] for _, row in df.iterrows()], dtype=float))


def snapshot_path(args: Any) -> Path | None:
    raw = str(getattr(args, "snapshot_dir", "")).strip()
    return Path(raw).expanduser() if raw else None


@dataclass(frozen=True)
class LooData:
    rows: list[dict[str, Any]]
    all_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_true: np.ndarray
    snapshot_dir: Path | None

    @property
    def n_points(self) -> int:
        return int(self.y_true.shape[0])


def load_loo_data(args: Any, *, empty_label: str, reset_index: bool = True) -> LooData:
    snap = snapshot_path(args)
    rows = build_rows_with_targets_source(root_folder_id=args.root_folder_id, env_file=Path(args.env_file), curve_points=args.curve_points, snapshot_dir=snap)
    all_df, train_df, test_df = select_train_test(rows, args)
    x_train, x_test = make_feature_matrix(train_df), make_feature_matrix(test_df)
    if reset_index:
        x_train, x_test = x_train.reset_index(drop=True), x_test.reset_index(drop=True)
    if x_train.empty:
        raise RuntimeError(f"Пустая матрица признаков для {empty_label}")
    y_train = target_matrix(train_df, args.curve_points)
    return LooData(rows, all_df, train_df, test_df, x_train, x_test, y_train, target_matrix(test_df, args.curve_points)[0], snap)


def curve_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    y_pred, y_true = clip01(y_pred), clip01(y_true)
    abs_err = np.abs(y_pred - y_true)
    d_pred, d_true = np.diff(y_pred), np.diff(y_true)
    dd_pred, dd_true = np.diff(y_pred, n=2), np.diff(y_true, n=2)
    spearman = float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman"))
    pearson = float(pd.Series(y_pred).corr(pd.Series(y_true), method="pearson"))
    return {
        "spearman": 0.0 if np.isnan(spearman) else spearman,
        "pearson": 0.0 if np.isnan(pearson) else pearson,
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mae": float(np.mean(abs_err)),
        "spike_rmse": float(np.sqrt(np.mean((d_pred - d_true) ** 2))) if d_pred.size else 0.0,
        "curvature_rmse": float(np.sqrt(np.mean((dd_pred - dd_true) ** 2))) if dd_pred.size else 0.0,
    }


def prediction_frame(y_pred: np.ndarray, y_true: np.ndarray, *, extra: dict[str, Any] | None = None) -> pd.DataFrame:
    y_pred, y_true = clip01(y_pred), clip01(y_true)
    data = {"point_idx": list(range(len(y_pred))), "pred_retention": y_pred, "pred_retention_norm": y_pred, "pred_score_raw": y_pred, "true_retention": y_true, "abs_error": np.abs(y_pred - y_true)}
    return pd.DataFrame((extra or {}) | data)


@dataclass(frozen=True)
class LooArtifacts:
    out_dir: Path
    dataset_name: str
    prediction_name: str = "holdout_prediction_vs_true.csv"

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_path(self) -> Path:
        return self.out_dir / self.dataset_name

    @property
    def prediction_path(self) -> Path:
        return self.out_dir / self.prediction_name

    @property
    def metrics_path(self) -> Path:
        return self.out_dir / "metrics.json"

    def write_tables(self, all_df: pd.DataFrame, result_df: pd.DataFrame) -> None:
        all_df.to_csv(self.dataset_path, index=False)
        result_df.to_csv(self.prediction_path, index=False)

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_items(rows: dict[str, Any]) -> str:
    return os.linesep.join(f"{key}: {value}" for key, value in rows.items())


def print_run_report(title: str, metrics: dict[str, Any], result_df: pd.DataFrame | None = None) -> None:
    print(title)
    print(metric_items(metrics))
    if result_df is not None:
        fmt = {"pred_retention": lambda x: f"{x:0.5f}", "true_retention": lambda x: f"{x:0.5f}", "abs_error": lambda x: f"{x:0.5f}"}
        print(f"Holdout prediction vs true ({len(result_df)} points)")
        print(result_df.to_string(index=False, formatters=fmt))
