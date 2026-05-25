import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.analysis.curve_fitting import fit_double_exp, fit_hill_curve, fit_weibull
from train.common.seq_data_utils import load_all_merged


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_best_curve(time_sec, retention):
    curves = {"hill": fit_hill_curve, "double_exp": fit_double_exp, "weibull": fit_weibull}
    best_mae = float("inf")
    best_type = "hill"
    best_params = np.zeros(5)

    for c_type, c_func in curves.items():
        pred_y, params = c_func(time_sec, retention)
        mae = np.mean(np.abs(retention - pred_y))
        if mae < best_mae:
            best_mae = mae
            best_type = c_type
            p_pad = np.zeros(5)
            p_pad[: len(params)] = params
            best_params = p_pad

    return {"hill": 0, "double_exp": 1, "weibull": 2}[best_type], best_params, best_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters-file", default="analysis/video_clustering/kmeans/clusters.json")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    cluster_map = {}
    if os.path.exists(args.clusters_file):
        with open(args.clusters_file, encoding="utf-8") as f:
            c_data = json.load(f)
            for vid, v_info in c_data.get("videos", {}).items():
                cluster_map[vid] = v_info.get("cluster_id", v_info.get("kmeans_cluster_id", -1))
        logger.info(f"Loaded clusters for {len(cluster_map)} videos")
    else:
        logger.warning(f"Clusters file not found: {args.clusters_file}")
    logger.info("Loading retention curves to fit parameters...")
    video_dfs = load_all_merged(args.output_dir, args.data_dir, use_curve_raw=True, emb_pca_components=0)

    curve_features = {}
    for vid, df in video_dfs.items():
        df_sorted = df.sort_values("time").reset_index(drop=True)
        t_sec = df_sorted["time"].values if pd.api.types.is_numeric_dtype(df_sorted["time"]) else np.arange(len(df_sorted))
        ret = df_sorted["retention"].values

        c_type, c_params, c_mae = get_best_curve(t_sec, ret)
        curve_features[vid] = {
            "best_curve_type": c_type,
            "curve_p0": c_params[0],
            "curve_p1": c_params[1],
            "curve_p2": c_params[2],
            "curve_p3": c_params[3],
            "curve_p4": c_params[4],
            "curve_fit_mae": c_mae,
        }
    logger.info(f"Computed curve parameters for {len(curve_features)} videos")
    csv_files = glob.glob(os.path.join(args.output_dir, "*_features.csv")) + glob.glob(os.path.join(args.output_dir, "*_features.csv.partial"))

    updated_count = 0
    for path in csv_files:
        vid = os.path.basename(path).replace("_features.csv.partial", "").replace("_features.csv", "")
        df = pd.read_csv(path, index_col=0 if pd.read_csv(path, nrows=0).columns[0] == "Unnamed: 0" else None)
        new_cols = {}
        if "video_cluster" in df.columns:
            df.drop(columns=["video_cluster"], inplace=True)

        c_id = cluster_map.get(vid, -1)
        new_cols["video_cluster"] = c_id
        for col in ["best_curve_type", "curve_p0", "curve_p1", "curve_p2", "curve_p3", "curve_p4", "curve_fit_mae"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        c_feats = curve_features.get(vid, {})
        for k, v in c_feats.items():
            new_cols[k] = v

        df = pd.concat([df, pd.DataFrame([new_cols] * len(df), index=df.index)], axis=1)
        df.to_csv(path)
        updated_count += 1

    logger.info(f"Updated {updated_count} files in {args.output_dir}")


if __name__ == "__main__":
    main()
