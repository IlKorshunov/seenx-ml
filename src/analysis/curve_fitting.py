"""Analytic retention-curve shapes (hill, double exponential, Weibull) shared by pred_curves, cluster features, and aggregator."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

C_FALLBACK_RATIO = 0.15


def hill_curve(x: np.ndarray, a, b, c, d):
    return d + (a - d) / (1.0 + np.power(x / c, b))


def double_exp_curve(x: np.ndarray, a, b, c, d, e):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e


def weibull_curve(x: np.ndarray, d, lam, k):
    return d * np.exp(-np.power(x / lam, k))

def fit_hill_curve(time_sec, retention):
    y, n = np.clip(retention, 0.0, 100.0), len(retention)
    c_max = C_FALLBACK_RATIO * n
    popt, _ = curve_fit( hill_curve, time_sec, y, p0=[float(y[-1]), 0.8, min(max(1.0, float(n * 0.20)), c_max), float(y[0])], bounds=([0, 0.01, 1, 0], [100, 20, c_max, 100]), maxfev=8000, )
    return hill_curve(time_sec, *popt), np.array(popt, dtype=float)

def fit_double_exp(time_sec, retention):
    y = np.clip(retention, 0.0, 100.0)
    drop = float(y[0] - y[-1])
    popt, _ = curve_fit(double_exp_curve, time_sec, y, p0=[drop * 0.6, 0.05, drop * 0.3, 0.005, float(y[-1])], bounds=([0, 1e-4, 0, 1e-5, 0], [100, 1.0, 100, 0.5, 100]), maxfev=8000, ) 
    return double_exp_curve(time_sec, *popt), np.array(popt, dtype=float)

def fit_weibull(time_sec, retention):
    y, n = np.clip(retention, 0.0, 100.0), len(time_sec)
    popt, _ = curve_fit( weibull_curve, time_sec + 1.0, y, p0=[float(y[0]), float(n * 0.3), 0.5], bounds=([0, 1, 0.01], [100, n * 2, 5.0]), maxfev=8000, )
    return weibull_curve(time_sec + 1.0, *popt), np.array(popt, dtype=float)
