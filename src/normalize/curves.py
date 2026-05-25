from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def clip_unit_interval(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=float), 0.0, 1.0)


def savgol_smooth(curve: np.ndarray, *, window: int = 7, order: int = 3) -> np.ndarray:
    return savgol_filter(np.asarray(curve, dtype=float), window_length=window, polyorder=order).astype(float)


def smooth_max_step(curve: np.ndarray, max_step: float) -> np.ndarray:
    out = np.asarray(curve, dtype=float).copy()
    step = float(max_step)
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        if abs(delta) > step:
            out[i] = out[i - 1] + step * np.sign(delta)
    return out


def smooth_curve_savgol_then_max_step(curve: np.ndarray, *, max_step: float = 0.05, savgol_window: int = 7, savgol_order: int = 3) -> np.ndarray:
    return smooth_max_step(savgol_smooth(curve, window=savgol_window, order=savgol_order), max_step)


def soft_non_increasing(curve: np.ndarray, *, max_increase: float) -> np.ndarray:
    out = np.asarray(curve, dtype=float).copy()
    inc = float(max(0.0, max_increase))
    for i in range(1, len(out)):
        if out[i] > out[i - 1]:
            out[i] = min(out[i], out[i - 1] + inc)
    return clip_unit_interval(out)
