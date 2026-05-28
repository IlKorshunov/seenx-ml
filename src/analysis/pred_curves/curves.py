from __future__ import annotations

import numpy as np

from ..curve_fitting import double_exp_curve as double_exp
from ..curve_fitting import fit_double_exp, fit_hill_curve, fit_weibull
from ..curve_fitting import hill_curve as hill
from ..curve_fitting import weibull_curve as weibull

N_POINTS = 100
MAX_PARAMS = 5
CURVE_DEFS = {"hill": {"names": ["a", "b", "c", "d"], "n": 4}, "double_exp": {"names": ["a", "b", "c", "d", "e"], "n": 5}, "weibull": {"names": ["d", "lam", "k"], "n": 3}}
_FUNCS = {"hill": hill, "double_exp": double_exp, "weibull": weibull}


def fit_curve(curve_type: str, retention: np.ndarray) -> np.ndarray | None:
    t = np.arange(len(retention), dtype=np.float64)
    fitters = {"hill": fit_hill_curve, "double_exp": fit_double_exp, "weibull": fit_weibull}
    _, params = fitters[curve_type](t, retention)
    return np.array(params, dtype=np.float64)

def reconstruct(curve_type: str, params: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    t = np.arange(n_points, dtype=np.float64)
    if curve_type == "weibull":
        return np.clip(weibull(t + 1.0, *params), 0, 100)
    return np.clip(_FUNCS[curve_type](t, *params), 0, 100)

def resample(curve: np.ndarray, target_len: int) -> np.ndarray:
    if len(curve) == target_len:
        return curve.copy()
    return np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(curve)), curve)
