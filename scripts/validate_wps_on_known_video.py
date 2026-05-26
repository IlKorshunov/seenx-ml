#!/usr/bin/env python3
"""
Validates WPS (words-per-second) extraction on a synthetic video with a
known transcript.

Methodology
-----------
1. A silent 30-second MP4 is generated with FFmpeg (no real speech needed).
2. A ground-truth transcript is provided manually — simulating what Whisper
   would return if the presenter had spoken the listed phrases at the given times.
3. The Whisper call inside `get_segments_and_duration` is replaced with a stub
   so the heavy GPU step is bypassed.
4. `extract_wps` runs exactly as in production; only the transcript source differs.
5. The resulting per-second WPS array is compared against the analytically
   computed expected values.

Ground-truth scenario (30-second educational video)
----------------------------------------------------
  0–10 s  – introduction : 11 words  → 1.10 wps
 10–25 s  – main content : 60 words  → 4.00 wps
 25–30 s  – outro        :  6 words  → 1.20 wps

Run
---
    python scripts/validate_wps_on_known_video.py
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── ground truth ───────────────────────────────────────────────────────────────

VIDEO_DURATION = 30  # seconds

KNOWN_TRANSCRIPT: list[dict] = [
    {
        "text": "hello everyone welcome to this video tutorial today we will learn",
        "start": 0.0,
        "end": 10.0,
    },  # 11 words / 10 s = 1.10 wps
    {
        "text": " ".join(f"concept{i}" for i in range(60)),
        "start": 10.0,
        "end": 25.0,
    },  # 60 words / 15 s = 4.00 wps
    {
        "text": "thanks for watching subscribe for more",
        "start": 25.0,
        "end": 30.0,
    },  # 6 words / 5 s  = 1.20 wps
]

EXPECTED_WPS = np.concatenate([
    np.full(10, 11 / 10.0),   # seconds  0–9
    np.full(15, 60 / 15.0),   # seconds 10–24
    np.full(5,   6 /  5.0),   # seconds 25–29
])


# ── helpers ────────────────────────────────────────────────────────────────────

def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_stubs() -> None:
    """Inject lightweight stubs for Whisper and GPU-dependent modules."""
    for pkg in ("src", "src.extractors", "src.extractors.text", "src.utils"):
        m = types.ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m

    seenx = types.ModuleType("src.seenx_utils")
    seenx.get_video_duration = lambda _p: float(VIDEO_DURATION)  # type: ignore[attr-defined]

    log = types.ModuleType("src.utils.logger")
    log.Logger = lambda show=True: types.SimpleNamespace(  # type: ignore[attr-defined]
        get_logger=lambda: types.SimpleNamespace(info=lambda *a, **k: None)
    )

    config = types.ModuleType("src.utils.config")
    config.Config = dict  # type: ignore[attr-defined]

    transcript = types.ModuleType("src.utils.transcript_cache")
    transcript.get_transcript = lambda _path, _cfg: {"segments": KNOWN_TRANSCRIPT}  # type: ignore[attr-defined]

    constants = types.ModuleType("src.extractors.text.constants")
    constants.WPS_COLS = {"wps"}  # type: ignore[attr-defined]

    for name, mod in {
        "src.seenx_utils": seenx,
        "src.utils.config": config,
        "src.utils.logger": log,
        "src.utils.transcript_cache": transcript,
        "src.extractors.text.constants": constants,
    }.items():
        sys.modules[name] = mod

    _load("src.extractors.text._base", "src/extractors/text/_base.py")


def _create_silent_video(path: str, duration: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=blue:size=128x72:rate=1:duration={duration}",
            "-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono:duration={duration}",
            "-shortest", path,
        ],
        check=True,
    )


# ── validation ─────────────────────────────────────────────────────────────────

def validate() -> bool:
    _install_stubs()
    extract_wps = _load(
        "src.extractors.text.wps_feature",
        "src/extractors/text/wps_feature.py",
    ).extract_wps

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name

    print(f"  Generating {VIDEO_DURATION}s silent video → {video_path}")
    _create_silent_video(video_path, VIDEO_DURATION)

    print("  Running extract_wps (Whisper stubbed with known transcript)…\n")
    df = extract_wps(video_path=video_path, config=None)
    actual = df["wps"].values

    # ── per-second report ──────────────────────────────────────────────────────
    seg_label = (
        ["intro  (0–9s)"] * 10
        + ["content (10–24s)"] * 15
        + ["outro  (25–29s)"] * 5
    )
    print(f"{'sec':>4}  {'segment':<20}  {'expected':>10}  {'actual':>10}  {'Δ':>10}  {'ok':>4}")
    print("─" * 66)
    all_ok = True
    for i, (exp, act, lbl) in enumerate(zip(EXPECTED_WPS, actual, seg_label)):
        delta = abs(exp - act)
        ok = delta < 1e-9
        if not ok:
            all_ok = False
        mark = "✓" if ok else "✗"
        print(f"{i:>4}  {lbl:<20}  {exp:>10.4f}  {act:>10.4f}  {delta:>10.2e}  {mark:>4}")

    print()
    if all_ok:
        print("RESULT: PASS — all 30 per-second WPS values match ground truth exactly")
    else:
        worst = int(np.argmax(np.abs(EXPECTED_WPS - actual)))
        print(
            f"RESULT: FAIL — largest deviation {np.abs(EXPECTED_WPS - actual).max():.2e}"
            f" at second {worst}"
        )

    # ── segment-level summary ─────────────────────────────────────────────────
    print()
    print("Segment-level summary:")
    slices = [(0, 10, "intro"), (10, 25, "content"), (25, 30, "outro")]
    for s, e, name in slices:
        measured = actual[s:e].mean()
        expected_mean = EXPECTED_WPS[s:e].mean()
        print(f"  {name:<10}  mean wps expected={expected_mean:.3f}  measured={measured:.3f}")

    return all_ok


if __name__ == "__main__":
    print("=" * 66)
    print("WPS Extraction Validation — Known Video")
    print("=" * 66)
    print()
    print("Ground-truth transcript:")
    for seg in KNOWN_TRANSCRIPT:
        nw = len(seg["text"].split())
        dur = seg["end"] - seg["start"]
        print(f"  [{seg['start']:.0f}–{seg['end']:.0f}s]  {nw} words / {dur:.0f}s = {nw/dur:.2f} wps")
    print()

    ok = validate()
    sys.exit(0 if ok else 1)
