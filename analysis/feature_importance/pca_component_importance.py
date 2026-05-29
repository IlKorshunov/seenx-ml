"""PCA component influence analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib  # type: ignore[import-not-found]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]
from sklearn.decomposition import PCA  # type: ignore[import-not-found]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]

from .utils import aggregate_per_video, default_output_dir, feature_group_of, load_all_videos, prepare_X_y


def _safe_corr(values: np.ndarray, target: np.ndarray) -> float:
    if len(values) < 2 or np.std(values) == 0 or np.std(target) == 0:
        return 0.0
    return float(np.corrcoef(values, target)[0, 1])


def compute_pca_component_importance(X: pd.DataFrame, y: pd.Series, n_components: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_comp = min(n_components or min(10, X.shape[0], X.shape[1]), X.shape[0], X.shape[1])
    if n_comp < 1:
        raise ValueError("PCA requires at least one sample and one feature")

    X_scaled = StandardScaler().fit_transform(X.values.astype(float))
    pca = PCA(n_components=n_comp, random_state=42)
    scores = pca.fit_transform(X_scaled)
    y_arr = y.values.astype(float)

    rows = []
    for idx in range(n_comp):
        corr = _safe_corr(scores[:, idx], y_arr)
        rows.append(
            {
                "component": f"PC{idx + 1}",
                "component_idx": idx,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[idx]),
                "target_correlation": corr,
                "importance": abs(corr) * float(pca.explained_variance_ratio_[idx]),
            }
        )
    component_df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)

    component_weights = component_df.sort_values("component_idx")["importance"].to_numpy()
    feature_scores = np.abs(pca.components_).T @ component_weights
    feature_df = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": feature_scores,
                "group": [feature_group_of(feature) for feature in X.columns],
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return component_df, feature_df


def _plot_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, out_path: Path, top_n: int) -> None:
    top = df.head(top_n)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.35)))
    ax.barh(top[label_col][::-1], top[value_col][::-1], color="steelblue")
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pca_component_importance(
    output_dir: str = "output", results_dir: str | None = None, target: str = "target_avg_retention", top_n: int = 30
) -> dict[str, pd.DataFrame]:
    if results_dir is None:
        results_root = default_output_dir()
    else:
        results_root = Path(results_dir)
    out_dir = results_root / "pca"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_dfs = load_all_videos(output_dir)
    agg_df = aggregate_per_video(video_dfs)
    X, y = prepare_X_y(agg_df, target=target)
    component_df, feature_df = compute_pca_component_importance(X, y, n_components=min(top_n, X.shape[0], X.shape[1]))

    component_df.to_csv(out_dir / "pca_component_importance.csv", index=False)
    feature_df.to_csv(out_dir / "pca_feature_importance.csv", index=False)
    _plot_bar(component_df, "component", "importance", f"PCA component influence - {target}", out_dir / "pca_component_importance.png", top_n)
    _plot_bar(feature_df, "feature", "importance", "PCA-derived feature influence", out_dir / "pca_feature_importance.png", top_n)
    return {"component_importance": component_df, "feature_importance": feature_df}
