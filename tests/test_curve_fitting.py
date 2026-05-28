import numpy as np

from src.analysis import curve_fitting as module


def test_curve_shapes_are_finite_and_monotone_like():
    x = np.array([0.0, 1.0, 2.0, 4.0], dtype=float)

    hill = module.hill_curve(x + 1.0, 90.0, 1.2, 2.0, 35.0)
    double = module.double_exp_curve(x, 40.0, 0.5, 30.0, 0.1, 20.0)
    weibull = module.weibull_curve(x + 1.0, 100.0, 10.0, 0.8)

    assert np.isfinite(hill).all()
    assert np.isfinite(double).all()
    assert np.isfinite(weibull).all()
    assert double[0] > double[-1]
    assert weibull[0] > weibull[-1]


def test_fitters_clip_inputs_and_return_curve_plus_params(monkeypatch):
    time_sec = np.arange(6, dtype=float)
    retention = np.array([130.0, 95.0, 80.0, 60.0, -5.0, 25.0], dtype=float)
    calls = []

    def fake_curve_fit(func, x, y, p0, bounds, maxfev):
        calls.append({"func": func.__name__, "x": x.copy(), "y": y.copy(), "p0": list(p0), "bounds": bounds, "maxfev": maxfev})
        if func is module.hill_curve:
            return np.array([90.0, 1.0, 2.0, 20.0]), None
        if func is module.double_exp_curve:
            return np.array([30.0, 0.2, 20.0, 0.05, 10.0]), None
        return np.array([90.0, 3.0, 0.7]), None

    monkeypatch.setattr(module, "curve_fit", fake_curve_fit)

    hill_y, hill_params = module.fit_hill_curve(time_sec, retention)
    double_y, double_params = module.fit_double_exp(time_sec, retention)
    weibull_y, weibull_params = module.fit_weibull(time_sec, retention)

    assert hill_y.shape == time_sec.shape
    assert double_y.shape == time_sec.shape
    assert weibull_y.shape == time_sec.shape
    np.testing.assert_allclose(hill_params, [90.0, 1.0, 2.0, 20.0])
    np.testing.assert_allclose(double_params, [30.0, 0.2, 20.0, 0.05, 10.0])
    np.testing.assert_allclose(weibull_params, [90.0, 3.0, 0.7])
    assert [call["func"] for call in calls] == ["hill_curve", "double_exp_curve", "weibull_curve"]
    assert all(call["maxfev"] == 8000 for call in calls)
    assert all(call["y"].min() >= 0.0 and call["y"].max() <= 100.0 for call in calls)
    np.testing.assert_allclose(calls[-1]["x"], time_sec + 1.0)
