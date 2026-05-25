from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.normalize.curves import clip_unit_interval


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value).strip())
    return cleaned or "unnamed"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_features_llm_payload(snapshot_dir: Path, video_folder: str) -> dict[str, Any]:
    video_dir = snapshot_dir / _safe_name(video_folder)
    if not video_dir.exists():
        return {}
    candidate = video_dir / "transcripts" / "features_llm.json"
    if candidate.exists():
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    for child in sorted(video_dir.iterdir()):
        if not child.is_dir():
            continue
        path = child / "features_llm.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def integration_strength_curve(payload: dict[str, Any], curve_points: int) -> np.ndarray:
    if not payload:
        return np.zeros((curve_points,), dtype=float)
    source = payload.get("source", {}) if isinstance(payload.get("source"), dict) else {}
    duration = _to_float(source.get("duration_seconds", 0.0), 0.0)
    text_features = payload.get("text_features", {}) if isinstance(payload.get("text_features"), dict) else {}
    integration = text_features.get("integration_feature", {}) if isinstance(text_features.get("integration_feature"), dict) else {}
    series = integration.get("series", {}) if isinstance(integration.get("series"), dict) else {}
    rows = series.get("integration_present", []) if isinstance(series.get("integration_present"), list) else []
    if not rows:
        return np.zeros((curve_points,), dtype=float)

    strengths = np.zeros((curve_points,), dtype=float)
    if duration > 1e-6:
        for i in range(curve_points):
            frac = i / max(1, curve_points - 1)
            t = frac * duration
            value = 0.0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                start = _to_float(row.get("start", 0.0), 0.0)
                end = _to_float(row.get("end", 0.0), 0.0)
                if end < start:
                    start, end = end, start
                if start <= t <= end:
                    value = _to_float(row.get("value", 0.0), 0.0)
                    break
            strengths[i] = np.clip(value, 0.0, 1.0)
        return strengths

    values: list[float] = []
    for row in rows:
        if isinstance(row, dict):
            values.append(np.clip(_to_float(row.get("value", 0.0), 0.0), 0.0, 1.0))
    if not values:
        return strengths
    if len(values) == 1:
        strengths[:] = values[0]
        return strengths
    x_old = np.linspace(0.0, 1.0, num=len(values))
    x_new = np.linspace(0.0, 1.0, num=curve_points)
    return clip_unit_interval(np.interp(x_new, x_old, np.asarray(values, dtype=float)))


def build_integration_matrix(df: pd.DataFrame, snapshot_dir: Path | None, curve_points: int) -> np.ndarray:
    out = np.zeros((len(df), curve_points), dtype=float)
    if snapshot_dir is None or not snapshot_dir.exists():
        return out
    for i in range(len(df)):
        vf = str(df.iloc[i].get("video_folder", "")).strip()
        payload = load_features_llm_payload(snapshot_dir, vf)
        out[i] = integration_strength_curve(payload, curve_points)
    return out


def compute_percentile_curves(y_train: np.ndarray, qs: tuple[int, ...] = (5, 10, 25, 50, 75, 90, 95)) -> np.ndarray:
    return np.stack([np.percentile(y_train, q, axis=0).astype(float) for q in qs])


def make_time_features(steps: int, n_sinusoidal: int = 4) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=float)
    cols = [t]
    for k in range(1, int(n_sinusoidal) + 1):
        cols.append(np.sin(2 * np.pi * k * t))
        cols.append(np.cos(2 * np.pi * k * t))
    return np.stack(cols, axis=1)


def knn_weighted_baseline(X_train: np.ndarray, X_query: np.ndarray, y_train: np.ndarray, *, k: int, temperature: float) -> np.ndarray:
    n_q = X_query.shape[0]
    kk = min(int(max(1, k)), X_train.shape[0])
    temp = float(max(temperature, 1e-8))
    out = np.zeros((n_q, y_train.shape[1]), dtype=float)
    for i in range(n_q):
        dist = np.linalg.norm(X_train - X_query[i], axis=1)
        idx = np.argpartition(dist, kk - 1)[:kk]
        sub = dist[idx]
        w = np.exp(-sub / temp)
        w = w / np.maximum(w.sum(), 1e-12)
        out[i] = w @ y_train[idx]
    return out[0] if n_q == 1 else out


def resolve_training_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_b = getattr(torch.backends, "mps", None)
        if mps_b is not None and mps_b.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)
