"""Retention diagnostics and advice.

The module finds low-retention / sharp-drop intervals in already prepared
feature CSV files and converts ranked feature signals into editing advice.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AdviceDirection = Literal["low", "high"]
AdviceRule = tuple[AdviceDirection, str]

class AdviceItem(TypedDict):
    feature: str
    value: float
    median: float
    advice: str


class DropCriteria(TypedDict):
    fixed_threshold: float
    percentile: float
    threshold: float


class SegmentAdvice(TypedDict):
    video_id: str
    start_sec: int
    end_sec: int
    retention_delta: float
    avg_derivative: float
    advice: list[AdviceItem]


NON_FEATURE_COLS = {
    "time",
    "time_sec",
    "retention",
    "audience_watch_ratio",
    "is_ad",
    "ad_segment_length",
    "is_ad_x_viewer_address",
}

ADVICE_RULES: dict[str, AdviceRule] = {
    "admiration": ("low", "добавить момент, вызывающий восхищение или сильную оценку"),
    "amusement": ("low", "добавить развлекательный или лёгкий эмоциональный фрагмент"),
    "anger": ("high", "смягчить конфликтный или раздражающий тон"),
    "annoyance": ("high", "убрать раздражающий повтор или спорную формулировку"),
    "approval": ("low", "усилить тезис, с которым зрителю легко согласиться"),
    "caring": ("low", "добавить более заботливую или поддерживающую подачу"),
    "confusion": ("high", "прояснить фрагмент и убрать неоднозначность"),
    "curiosity": ("low", "усилить интригу и ожидание ответа"),
    "desire": ("low", "сильнее показать ценность или желаемый результат"),
    "disappointment": ("high", "убрать разочаровывающий или проседающий по энергии фрагмент"),
    "disapproval": ("high", "смягчить резкую оценку или спорную позицию"),
    "disgust": ("high", "проверить отталкивающий визуальный или смысловой элемент"),
    "embarrassment": ("high", "убрать неловкий или сбивающий тональность момент"),
    "excitement": ("low", "добавить более эмоциональный и энергичный акцент"),
    "fear": ("high", "смягчить тревожный фрагмент или объяснить его контекст"),
    "gratitude": ("low", "добавить позитивную обратную связь или благодарность"),
    "grief": ("high", "сократить слишком тяжёлый эмоциональный участок"),
    "happy": ("low", "добавить более позитивную мимику или визуальную эмоцию"),
    "laughter_prob": ("low", "добавить более лёгкий или юмористический фрагмент"),
    "joy": ("low", "усилить позитивную эмоцию в подаче"),
    "love": ("low", "добавить более тёплую эмоциональную формулировку"),
    "nervousness": ("high", "сделать речь увереннее и спокойнее"),
    "optimism": ("low", "усилить позитивную перспективу или обещание пользы"),
    "pride": ("low", "добавить сильный результат или достижение"),
    "realization": ("low", "добавить момент инсайта или понятного вывода"),
    "relief": ("low", "добавить снятие напряжения или понятное решение проблемы"),
    "remorse": ("high", "сократить тяжёлый или чрезмерно извиняющийся фрагмент"),
    "sad": ("high", "разбавить грустную мимику или изменить эмоциональный тон"),
    "sadness": ("high", "разбавить грустный эмоциональный участок"),
    "surprise": ("low", "добавить неожиданный поворот или визуальный акцент"),
    "neutral": ("high", "снизить монотонность и добавить эмоциональный акцент"),
    "ekman_joy": ("low", "усилить позитивную эмоциональную подачу"),
    "ekman_excitement": ("low", "добавить больше энергии и эмоционального подъёма"),
    "ekman_sadness": ("high", "разбавить грустный фрагмент более активной подачей"),
    "ekman_neutral": ("high", "снизить нейтральность подачи"),
    "ekman_intensity": ("low", "усилить выраженность эмоции"),
    "has_background_music": ("low", "рассмотреть фоновую музыку или музыкальный акцент"),
    "music_changed": ("low", "добавить смену музыкального рисунка"),
    "music_only": ("high", "проверить, не вытесняет ли музыка речь или смысловой фрагмент"),
    "music_rms": ("low", "усилить музыкальную подложку или добавить звуковой акцент"),
    "music_zcr": ("low", "добавить более выразительную музыкальную фактуру"),
    "music_centroid": ("low", "сделать музыкальную подложку ярче по тембру"),
    "music_rolloff": ("low", "добавить более заметный верхний спектральный акцент"),
    "vocal_rms": ("low", "усилить голосовую дорожку относительно фона"),
    "vocal_zcr": ("high", "проверить резкость или шумность голосового фрагмента"),
    "vocal_centroid": ("low", "сделать голос более читаемым и ярким"),
    "vocal_rolloff": ("low", "проверить спектральную читаемость речи"),
    "beat_sync": ("low", "синхронизировать монтаж с ритмом музыки"),
    "beat_sync_ratio": ("low", "добавить монтажный акцент в музыкальные доли"),
    "viewer_address": ("low", "добавить прямое обращение к зрителю"),
    "hook_score": ("low", "усилить хук или обещание ценности"),
    "hook_has_address": ("low", "сформулировать обращение к зрителю в начале фрагмента"),
    "is_question": ("low", "добавить вопрос или интригу для вовлечения"),
    "question_density": ("low", "добавить интерактивный вопрос к аудитории"),
    "semantic_novelty": ("low", "добавить новый смысловой поворот или пример"),
    "topic_shift": ("low", "добавить более явный переход к новой мысли"),
    "hook_similarity": ("low", "связать участок с исходным обещанием ролика"),
    "global_topic_dist": ("high", "убрать слишком сильное отклонение от основной темы"),
    "semantic_momentum": ("low", "усилить развитие темы и связность повествования"),
    "segment_self_similarity": ("high", "сократить повторяющийся фрагмент или добавить новую информацию"),
    "curiosity_gap": ("low", "создать ожидание ответа или открытый информационный разрыв"),
    "storytelling": ("low", "добавить элемент истории"),
    "viewer_engagement": ("low", "усилить вовлекающий призыв или интерактивный элемент"),
    "has_example": ("low", "добавить конкретный пример"),
    "information_density": ("low", "добавить больше полезной информации на участке"),
    "cumulative_info": ("low", "ускорить подачу новых фактов или тезисов"),
    "narrative_momentum": ("low", "усилить движение истории или логики объяснения"),
    "content_rhythm": ("low", "выровнять ритм подачи контента"),
    "engagement_surprise": ("low", "добавить неожиданный вовлекающий элемент"),
    "visual_audio_sync": ("low", "синхронизировать визуальные изменения со звуковыми акцентами"),
    "edit_pace": ("low", "ускорить монтаж или добавить смену плана"),
    "scene_novelty": ("low", "добавить смену сцены или визуальный поворот"),
    "short_insert": ("low", "добавить короткую вставку для удержания внимания"),
    "short_insert_rate": ("low", "увеличить частоту коротких вставок"),
    "motion_speed": ("low", "добавить визуальную динамику"),
    "flow_mag_med": ("low", "добавить движение камеры или объекта"),
    "radial_med": ("low", "добавить zoom-in или визуальное приближение"),
    "radial_ratio": ("low", "усилить согласованное радиальное движение в кадре"),
    "speaker_prob": ("low", "вернуть автора/спикера в кадр"),
    "face_screen_ratio": ("low", "увеличить присутствие лица или крупность плана"),
    "faces_total_ratio": ("low", "увеличить долю кадров с лицом или реакцией человека"),
    "face_area_ratio": ("low", "сделать лицо заметнее в кадре"),
    "text_prob": ("low", "добавить читаемую подпись или визуальный тезис"),
    "screencast_prob": ("high", "разбавить демонстрацию экрана живым планом или вставкой"),
    "overlay_prob": ("low", "добавить визуальный оверлей, схему или акцент"),
    "object_count": ("low", "добавить визуальные объекты или предметные примеры"),
    "unique_classes": ("low", "добавить больше визуального разнообразия объектов"),
    "bumper_score": ("high", "сократить служебную заставку или перебивку"),
    "visual_entropy": ("low", "добавить визуальную насыщенность без перегруза"),
    "visual_complexity": ("high", "упростить перегруженный кадр"),
    "visual_complexity_gradient": ("high", "сгладить резкий рост визуальной сложности"),
    "visual_complexity_acceleration": ("high", "убрать слишком резкое усложнение кадра"),
    "brightness": ("low", "проверить яркость и читаемость кадра"),
    "sharpness": ("low", "проверить резкость изображения"),
    "color_temperature": ("low", "сделать цветовую температуру более выразительной"),
    "color_saturation": ("low", "усилить цветовой акцент"),
    "rms": ("low", "проверить громкость и выразительность звука"),
    "zcr": ("high", "проверить шумность или резкость аудио"),
    "centroid": ("low", "сделать звук более ясным по тембру"),
    "rolloff": ("low", "добавить спектральную выразительность звука"),
    "loudness_change": ("low", "добавить динамику громкости или звуковой акцент"),
    "loudness_variance": ("low", "сделать звуковую подачу менее монотонной"),
    "spectral_flux": ("low", "добавить звуковое событие или смену фактуры"),
    "sfx_energy": ("low", "добавить аккуратный звуковой эффект"),
    "speech_ratio": ("low", "добавить голосовое пояснение или убрать длинную немую паузу"),
    "silence_stretch": ("high", "сократить длительную тишину"),
    "wps": ("low", "ускорить темп речи или добавить содержательный комментарий"),
    "pitch_mean": ("low", "сделать интонацию более выразительной"),
    "pitch_std": ("low", "добавить интонационное разнообразие"),
    "voiced_frac": ("low", "уменьшить долю пауз и немых участков"),
    "speech_rate_cv": ("high", "сделать темп речи ровнее"),
    "speech_complexity": ("high", "упростить объяснение или разбить мысль на части"),
    "lexical_diversity": ("low", "добавить более выразительную лексику"),
    "avg_word_length": ("high", "заменить сложные формулировки более простыми"),
    "syntactic_depth": ("high", "сделать речь короче и проще"),
    "pause_rate": ("high", "сократить затянутые паузы"),
    "crutch_cnt": ("high", "сократить слова-паразиты и повторы"),
    "speech_predictability": ("high", "добавить менее шаблонную формулировку"),
    "speech_lm_surprisal": ("high", "упростить неожиданно сложный речевой фрагмент"),
    "speech_lm_surprisal_vel": ("high", "сгладить резкий рост сложности речи"),
    "speech_intelligibility": ("low", "улучшить разборчивость речи"),
    "speech_mumble_index": ("high", "перезаписать или почистить неразборчивую речь"),
    "has_person_mention": ("low", "добавить персональный пример или упоминание человека"),
    "has_org_mention": ("low", "добавить конкретный бренд, компанию или контекст"),
    "chapter_id": ("high", "проверить структуру текущей главы ролика"),
    "n_chapters": ("low", "разбить ролик на более понятные смысловые блоки"),
    "topic_change_rate": ("high", "снизить слишком частые переходы между темами"),
    "desc_chapter_start": ("low", "добавить более явное начало смыслового блока"),
    "desc_chapter_boundary_dist": ("high", "перенести важный момент ближе к границе главы"),
    "timecode_like_weighted_30s": ("low", "добавить более заметный ориентир или таймкодный повод"),
    "comment_density_30s": ("low", "усилить фрагмент, который может провоцировать обсуждение"),
    "comment_question_rate_30s": ("low", "добавить вопрос, на который зрители захотят ответить"),
    "comment_positive_rate_30s": ("low", "усилить позитивный или полезный момент"),
    "comment_reply_depth_30s": ("low", "добавить спорный или обсуждаемый тезис для цепочки ответов"),
    "comment_aggression_rate_30s": ("high", "проверить спорный или конфликтный фрагмент"),
    "author_reply_rate_video": ("low", "усилить повод для ответа автора или продолжения обсуждения"),
    "avg_comment_length_video": ("low", "добавить более содержательный повод для комментариев"),
    "complex_words_ratio_video": ("high", "упростить лексику, если она снижает вовлечение"),
    "is_intro": ("high", "сократить затянутый вступительный участок"),
    "is_outro": ("high", "сократить концовку или раньше дать ключевое действие"),
    "edit_pace_x_screencast": ("high", "ускорить демонстрацию экрана или добавить смену визуала"),
    "title_transcript_gap": ("high", "сильнее связать содержание участка с обещанием в заголовке"),
    "title_delivery_30s": ("low", "быстрее выполнить обещание из заголовка"),
    "title_claim_intensity": ("high", "снизить чрезмерность обещания или быстрее его подтвердить"),
    "topic_sharpness_0_100": ("low", "сделать тезис участка более сфокусированным"),
}


def _video_id(path: Path) -> str:
    return path.stem.replace("_features", "")


def _load_ranked_features(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        return []
    if "avg_rank" in df.columns:
        df = df.sort_values("avg_rank")
    return [str(x) for x in df["feature"].dropna().tolist()]


def _find_prediction(predictions_root: Path | None, video_id: str) -> pd.DataFrame:
    if predictions_root is None or not predictions_root.exists():
        raise FileNotFoundError("Predictions root is required for retention advice")
    for path in sorted(predictions_root.rglob("holdout_prediction_vs_true.csv")):
        df = pd.read_csv(path)
        if "video" in df.columns:
            rows = df[df["video"].astype(str) == video_id]
            if not rows.empty:
                return rows.reset_index(drop=True)
        elif len(df):
            return df.reset_index(drop=True)
    raise FileNotFoundError(f"No holdout prediction found for video '{video_id}'")


def _signal(pred_df: pd.DataFrame) -> np.ndarray:
    for col in ("pred_retention", "predicted", "pred"):
        if col in pred_df.columns:
            return pd.to_numeric(pred_df[col], errors="coerce").interpolate().bfill().ffill().to_numpy(float)
    raise ValueError("Prediction file must contain pred_retention, predicted, or pred column")


def _merge_segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out, start = [], None
    for i, flag in enumerate(mask.tolist() + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def _find_segments(y: np.ndarray, window: int, min_len: int, drop_threshold: float, drop_percentile: float) -> list[tuple[int, int, float, float, float]]:
    if len(y) < max(window + 1, min_len):
        return []
    s = pd.Series(y)
    derivative = s.diff().rolling(window=window, min_periods=1).mean()
    percentile_threshold = float(np.nanpercentile(derivative.dropna(), drop_percentile))
    combined_threshold = min(drop_threshold, percentile_threshold)
    mask = (derivative <= combined_threshold).fillna(False).to_numpy()
    segments = []
    for start, end in _merge_segments(mask, min_len):
        seg_derivative = derivative.iloc[start : end + 1].dropna()
        segments.append(
            (
                start,
                end,
                float(seg_derivative.mean()),
                percentile_threshold,
                combined_threshold,
            )
        )
    return segments


def _feature_cols(df: pd.DataFrame, ranked: list[str], top_n: int) -> list[str]:
    numeric = [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    ranked_present = [c for c in ranked if c in numeric]
    rest = [c for c in numeric if c not in ranked_present]
    return (ranked_present + rest)[: max(top_n * 8, top_n)]


def _advice_for_segment(
    df: pd.DataFrame,
    start: int,
    end: int,
    ranked: list[str],
    top_n: int,
) -> list[AdviceItem]:
    cols = _feature_cols(df, ranked, top_n)
    candidates: list[tuple[float, AdviceItem]] = []
    for col in cols:
        if col not in ADVICE_RULES:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        seg_mean = float(vals.iloc[start : end + 1].mean())
        med = float(vals.median())
        q15, q85 = float(vals.quantile(0.15)), float(vals.quantile(0.85))
        direction, message = ADVICE_RULES[col]
        triggered = (direction == "low" and seg_mean <= q15) or (direction == "high" and seg_mean >= q85)
        if triggered:
            spread = max(abs(q85 - q15), 1e-9)
            severity = abs(seg_mean - med) / spread
            candidates.append(
                (
                    severity,
                    {
                        "feature": col,
                        "value": round(seg_mean, 4),
                        "median": round(med, 4),
                        "advice": message,
                    },
                )
            )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in candidates[:top_n]]


def _plot(video_id: str, y: np.ndarray, segments: list[SegmentAdvice], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(y)), y, label="retention / prediction", color="#1565C0", linewidth=2)
    for seg in segments:
        ax.axvspan(seg["start_sec"], seg["end_sec"], color="#C62828", alpha=0.18)
    ax.set_title(f"Retention diagnostics: {video_id}")
    ax.set_xlabel("second")
    ax.set_ylabel("retention")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_video(path: Path, ranked: list[str], args: Namespace) -> list[SegmentAdvice]:
    video_id = _video_id(path)
    df = pd.read_csv(path)
    pred_df = _find_prediction(Path(args.predictions_root) if args.predictions_root else None, video_id)
    y = _signal(pred_df)
    segments: list[SegmentAdvice] = []
    drop_criteria: DropCriteria | None = None
    for start, end, avg_derivative, percentile_threshold, effective_threshold in _find_segments(
        y,
        args.window,
        args.min_len,
        args.drop_threshold,
        args.drop_percentile,
    ):
        drop_criteria = {
            "fixed_threshold": args.drop_threshold,
            "percentile": args.drop_percentile,
            "threshold": round(effective_threshold, 4),
        }
        advice = _advice_for_segment(df, start, min(end, len(df) - 1), ranked, args.top_n)
        if advice:
            segments.append(
                {
                    "video_id": video_id,
                    "start_sec": start,
                    "end_sec": end,
                    "retention_delta": round(float(y[end] - y[start]), 4),
                    "avg_derivative": round(avg_derivative, 4),
                    "advice": advice,
                }
            )
    out_dir = Path(args.output_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if drop_criteria is None:
        drop_criteria = {
            "fixed_threshold": args.drop_threshold,
            "percentile": args.drop_percentile,
            "threshold": args.drop_threshold,
        }
    (out_dir / "advice.json").write_text(json.dumps({"video_id": video_id, "drop_criteria": drop_criteria, "segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{**{k: v for k, v in s.items() if k != "advice"}, "top_advice": "; ".join(a["advice"] for a in s["advice"])} for s in segments]).to_csv(out_dir / "segments.csv", index=False)
    _plot(video_id, y, segments, out_dir / "retention_advice.png")
    return segments


def run(args: Namespace) -> None:
    ranked = _load_ranked_features(Path(args.importance_path))
    all_segments = []
    for path in sorted(Path(args.features_dir).glob("*_features.csv")):
        all_segments.extend(analyze_video(path, ranked, args))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{k: v for k, v in s.items() if k != "advice"}, "n_advice": len(s["advice"])} for s in all_segments]).to_csv(out / "summary.csv", index=False)
    print(f"[retention_advice] videos={len(list(Path(args.features_dir).glob('*_features.csv')))} segments={len(all_segments)} -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Find retention drops and generate feature-based advice")
    p.add_argument("--features_dir", default="output")
    p.add_argument("--predictions_root", default="", help="Root with neural holdout_prediction_vs_true.csv files")
    p.add_argument("--importance_path", default="analysis/feature_importance/results/master_ranking.csv")
    p.add_argument("--output_dir", default="my_metrics/retention_advice")
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--min_len", type=int, default=5)
    p.add_argument("--drop_threshold", type=float, default=-0.3)
    p.add_argument("--drop_percentile", type=float, default=15.0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
