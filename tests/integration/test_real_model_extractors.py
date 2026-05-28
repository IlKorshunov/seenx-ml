from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


DATA_DIR = Path("data") / "wWtq4RlbRos"


def _require_data() -> Path:
    if not (DATA_DIR / "video.mp4").is_file() or not (DATA_DIR / "transcripts" / "whisper_segments.csv").is_file():
        pytest.skip("real data/transcript fixtures are missing")
    return DATA_DIR


def _cuda_or_cpu() -> str:
    torch = pytest.importorskip("torch")
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tiny_real_clip(tmp_path_factory) -> Path:
    data_dir = _require_data()
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to prepare model extractor clips")
    out_dir = tmp_path_factory.mktemp("real_model_clip")
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
            "fps=2,scale=224:-2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def _install_transcript_cache(video_path: Path, max_segments: int = 3) -> None:
    from src.utils import transcript_cache

    segments = []
    with (DATA_DIR / "transcripts" / "whisper_segments.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            segments.append({"start": float(row["start"]), "end": float(row["end"]), "text": row["text"]})
            if len(segments) >= max_segments:
                break
    transcript_cache._cache[os.path.abspath(str(video_path))] = {"segments": segments}


def test_clap_and_speech_emotion_models_on_tiny_real_clip(tiny_real_clip, tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from src.extractors.audio import clap_embedding_feature as clap
    from src.extractors.audio.speech_emotion_feature import SPEECH_EMOTION_COLS, extract_speech_emotion

    monkeypatch.setattr(clap, "embeddings_dir", lambda _video_path: str(tmp_path / "audio_embeddings"))
    config = {"device": _cuda_or_cpu()}

    try:
        clap_df = clap.extract_clap_embeddings(str(tiny_real_clip), config=config)
        emotion_df = extract_speech_emotion(str(tiny_real_clip), config=config)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"local audio model smoke skipped: {exc}")

    assert set(clap.CLAP_AUDIO_COLS).issubset(clap_df.columns)
    assert len(clap_df) >= 2
    assert np.isfinite(clap_df.to_numpy()).all()
    assert SPEECH_EMOTION_COLS.issubset(emotion_df.columns)
    assert np.isfinite(emotion_df.to_numpy()).all()


def test_clip_aesthetic_model_on_tiny_real_clip(tiny_real_clip):
    pytest.importorskip("transformers")
    from src.extractors.video.aesthetic_score_feature import extract_aesthetic_score

    try:
        df = extract_aesthetic_score(str(tiny_real_clip), {"device": _cuda_or_cpu()})
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"local CLIP aesthetic smoke skipped: {exc}")

    assert list(df.columns) == ["aesthetic_score"]
    assert len(df) >= 2
    assert df["aesthetic_score"].between(0, 10).all()


def test_user2_and_roberta_text_models_on_cached_real_transcript(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    video_path = _require_data() / "video.mp4"
    _install_transcript_cache(video_path)

    from src.extractors.text import semantic_embedding_feature as semantic
    from src.extractors.text.text_sentiment_feature import TEXT_SENTIMENT_COLS, extract_text_sentiment

    monkeypatch.setattr(semantic, "EMBEDDINGS_ROOT", str(tmp_path / "semantic_embeddings"))
    config = {"device": _cuda_or_cpu()}

    try:
        semantic_df = semantic.extract_semantic_embeddings(str(video_path), config=config)
        sentiment_df = extract_text_sentiment(str(video_path), config=config)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"local text model smoke skipped: {exc}")

    assert set(semantic._COLS).issubset(semantic_df.columns)
    assert np.isfinite(semantic_df.to_numpy()).all()
    assert TEXT_SENTIMENT_COLS.issubset(sentiment_df.columns)
    assert np.isfinite(sentiment_df.to_numpy()).all()
