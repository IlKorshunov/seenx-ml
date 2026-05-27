"""Integration smoke tests backed by local data and cached models."""

from __future__ import annotations

import json
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


DATA_ID = "wWtq4RlbRos"
DATA_DIR = Path("data") / DATA_ID


@pytest.fixture(autouse=True)
def restore_real_imports():
    for module_name in list(sys.modules):
        if module_name == "src" or module_name.startswith("src."):
            sys.modules.pop(module_name, None)
        if module_name == "tqdm" or module_name.startswith("tqdm."):
            sys.modules.pop(module_name, None)


def _require_real_data() -> Path:
    required = ("video.mp4", "audio.mp3", "retention.csv", "features_llm.json")
    if not all((DATA_DIR / name).is_file() for name in required):
        pytest.skip(f"real test data is missing under {DATA_DIR}")
    return DATA_DIR


@pytest.fixture(scope="module")
def tiny_real_video(tmp_path_factory) -> Path:
    data_dir = _require_real_data()
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to prepare a tiny real-data clip")

    out_dir = tmp_path_factory.mktemp("real_video")
    out_path = out_dir / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(data_dir / "video.mp4"),
            "-t",
            "3",
            "-vf",
            "fps=3,scale=160:-2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-ar",
            "22050",
            "-ac",
            "1",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def test_retention_loader_uses_real_video_csv_and_llm_features():
    data_dir = _require_real_data()

    from src.aggregator import get_retention

    retention = get_retention(str(data_dir / "video.mp4"), str(data_dir / "retention.csv"))
    llm_features = json.loads((data_dir / "features_llm.json").read_text())

    assert retention.index.name == "time"
    assert retention.shape[0] > 100
    assert retention["retention"].between(0, 200).all()
    assert len(llm_features) > 0


def test_lightweight_extractors_process_tiny_real_audio_video(tiny_real_video):
    from src.extractors.audio.loudness_dynamics_feature import extract_loudness_dynamics
    from src.extractors.video.color_features import ColorFeature
    from src.extractors.video.frame_feature import FrameQualityFeature

    audio_features = extract_loudness_dynamics(str(tiny_real_video), config={})
    assert list(audio_features.columns) == ["loudness_change", "loudness_variance"]
    assert len(audio_features) >= 2
    assert np.isfinite(audio_features.to_numpy()).all()

    frame_data = pd.DataFrame(index=range(12))
    context = {"data": frame_data}
    FrameQualityFeature(config={}).run(str(tiny_real_video), context)
    ColorFeature(config={}).run(str(tiny_real_video), context)

    expected_cols = {"brightness", "sharpness", "visual_complexity", "color_temperature", "color_saturation"}
    assert expected_cols.issubset(frame_data.columns)
    assert frame_data[list(expected_cols)].notna().sum().min() >= 3


def test_videomae_cached_model_extracts_embeddings_on_tiny_real_video(tiny_real_video, tmp_path):
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    from src.extractors.video.videomae_feature import extract_videomae_embeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = extract_videomae_embeddings(
        str(tiny_real_video),
        {"device": device},
        embeddings_root=str(tmp_path / "embeddings"),
        force=True,
    )

    assert embeddings.shape == (3, 768)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert float(np.abs(embeddings).sum()) > 0.0
