logger = Logger(show=True).get_logger()

CHECKPOINT_SUFFIX = ".partial"
FAILED_SUFFIX = ".failed_features.json"


def _checkpoint_path(output_path: str) -> str:
    return output_path + CHECKPOINT_SUFFIX

def _save_checkpoint(df: pd.DataFrame, output_path: str):
    df.to_csv(_checkpoint_path(output_path), index=True)

def _load_checkpoint(output_path: str) -> tuple[pd.DataFrame, list[str]]:
    path = _checkpoint_path(output_path)
    if not os.path.exists(path):
        return pd.DataFrame(), []
    df = pd.read_csv(path, index_col=0)
    return df, df.columns.tolist()


# Группы признаков, ожидаемые в итоговом результате. Здесь приведены
# представительные подмножества;
_VISUAL_COLS = {"brightness", "sharpness", "speaker_prob", "motion_speed",
                "edit_pace", "scene_novelty", "screencast_prob", }
_AUDIO_COLS = {"rms", "zcr", "centroid", "rolloff", "pitch_mean",
               "voiced_frac", "speech_rate_cv", }
_TEXT_COLS = {"wps", "viewer_address", "hook_score", "is_question",
              "semantic_novelty", "topic_shift", }
_COMMENT_COLS = {"comment_density_30s", "comment_question_rate_30s",
                 "author_reply_rate_video", }
_EMOTION_COLS = {"sent_joy", "sent_anger", "sent_surprise", }

_ALL_EXPECTED = ({"retention"} | _VISUAL_COLS | _AUDIO_COLS | _TEXT_COLS
                 | _COMMENT_COLS | _EMOTION_COLS | )  # остальные группы аналогичны


def _should_run(step_cols: set[str], only: set[str]|None, existing: set[str]) -> bool:
    """Запускать этап, если есть недостающие столбцы либо явное требование
    пересчёта через параметр only."""
    missing = step_cols - existing
    want_overwrite = only and (step_cols & only & existing)
    if not missing and not want_overwrite: return False
    return only is None or bool(missing & only) or bool(want_overwrite)


def get_retention(video_path: str, retention_csv_path: str) -> pd.DataFrame:
    video_duration = get_video_duration(video_path)
    csv_df = pd.read_csv(retention_csv_path)
    n_points = int(video_duration) + 1
    retention_pct = np.interp(
        np.linspace(0, 1, n_points),
        csv_df["time_ratio"].values,
        csv_df["audience_watch_ratio"].values * 100.0,
    )
    return pd.DataFrame(
        {"retention": retention_pct},
        index=pd.to_timedelta(np.arange(n_points), unit="s"),
    )


def _map_to_index(index, features: pd.DataFrame) -> pd.DataFrame:
    """Приведение значений признаков к единой временной сетке."""
    return pd.DataFrame(
        {col: np.interp(np.linspace(0, len(features) - 1, len(index)),
                        np.arange(len(features)),
                        features[col].astype("float64").values)
         for col in features.columns},
        index=index,
    )


def aggregate(video_path: str, audio_path: str, output_path: str,
              config: Config, retention_csv_path: str,
              data_dir: str = "data", only: set[str] | None = None,
              skip_comment_features: bool = False,
              skip_emotion_features: bool = False):
    accumulated, existing = _load_checkpoint(output_path)
    existing_set = set(existing)

    # Контроль качества ранее извлечённых столбцов: пустые, постоянные
    # и нулевые удаляются для повторного извлечения.
    bad_cols = [c for c in accumulated.columns
                if pd.api.types.is_numeric_dtype(accumulated[c])
                and (accumulated[c].dropna().eq(0).all()
                     or float(accumulated[c].std()) < 1e-9)]
    if "speaker_prob" in accumulated.columns:
        zero_ratio = float((accumulated["speaker_prob"] == 0).sum()) / max(len(accumulated), 1)
        if zero_ratio > 0.5:
            bad_cols += [c for c in ("speaker_prob", "face_screen_ratio",
                                      "faces_total_ratio", "face_area_ratio")
                         if c in accumulated.columns]
    if bad_cols:
        accumulated = accumulated.drop(columns=bad_cols, errors="ignore")
        existing_set = set(accumulated.columns.tolist())

    if _ALL_EXPECTED.issubset(existing_set) and only is None:
        return accumulated  # все признаки уже извлечены

    if "retention" not in existing_set:
        retention = get_retention(video_path, retention_csv_path)
    else:
        retention = accumulated[["retention"]].copy()

    if accumulated.empty:
        accumulated = retention.copy()

    def add_to_accumulated(mapped: pd.DataFrame) -> None:
        """Добавляет новые столбцы; при наличии параметра only дополни-тельно
        перезаписывает уже существующие столбцы, попавшие в область only."""
        nonlocal accumulated, existing_set
        new_cols = [c for c in mapped.columns if c not in accumulat-ed.columns]
        overwrite_cols = [c for c in mapped.columns
                          if c in accumulated.columns and only and c in only]
        if new_cols:
            accumulated = pd.concat([accumulated, mapped[new_cols]], ax-is=1)
        if overwrite_cols:
            accumulated[overwrite_cols] = mapped[overwrite_cols]
        if new_cols or overwrite_cols:
            existing_set = set(accumulated.columns.tolist())
            _save_checkpoint(accumulated, output_path)

    # Визуальные признаки на основе предобученных моделей кадрового анали-за
    if _should_run(_VISUAL_COLS, only, existing_set):
        passes = _build_visual_passes(config, data_dir)
        features = run_feature_pipeline(video_path, config, passes=passes,
                                        existing_features=existing_set)
        add_to_accumulated(_map_to_index(retention.index, features))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Извлечение признаков всех остальных групп. Каждый этап представлен
    # парой (множество ожидаемых столбцов, функция извлечения).
    _STEPS = [
        ("prosody",
         {"pitch_mean", "pitch_std", "voiced_frac", "speech_rate_cv", "pause_rate"},
         lambda: extract_prosody(video_path, config, existing)),
        ("hook score",
         {"hook_score", "hook_has_address", "is_question"},
         lambda: extract_hook_score(video_path, config, existing)),
        ("semantic embeddings",
         {"semantic_novelty", "topic_shift", "hook_similarity", },
         lambda: extract_semantic_embeddings(video_path, config, exist-ing)),
        ("sentiment", _EMOTION_COLS,
         lambda: extract_text_sentiment(video_path, config, existing)),
        ("comment features", _COMMENT_COLS,
         lambda: extract_comment_features(video_path, config, existing)),
        # Аналогично подключены извлекатели для прочих групп признаков:
        # темпа речи, рекламных сегментов, текстовой сложности,
        # кластеризации на главы и других.
    ]

    for name, cols, fn in _STEPS:
        # Опциональное отключение тяжёлых групп через флаги командной строки.
        if skip_comment_features and name == "comment features": continue
        if skip_emotion_features and name == "sentiment": continue
        if not _should_run(cols, only, existing_set): continue
        result = fn()
        if result is not None and not result.empty:
            add_to_accumulated(_map_to_index(retention.index, result))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    clear_transcript_cache()

    # Приведение эмоциональных характеристик к единому пространству
    if _RAW_EMOTION_COLS & set(accumulated.columns):
        emotions = compute_emotions_fusion(accumulated)
        for col in emotions.columns: accumulated[col] = emotions[col]
        accumulated = accumulated.drop(columns=list(_RAW_EMOTION_COLS & set(accumulated.columns)), errors="ignore")

    # Удаление устаревших и избыточных столбцов
    accumulated = accumulated.drop(columns=list(_REDUNDANT_COLS & set(accumulated.columns)), errors="ignore")
    return accumulated  


def aggregate_batch(data_dir: str, output_dir: str, config: Config):
    """Пакетная обработка нескольких роликов.
    Для каждой группы извлекателей последовательно обрабатываются все ро-лики,
    после чего нейросетевые модели группы выгружаются из памяти. Это поз-воляет
    загружать каждую тяжёлую модель в видеопамять однократно за весь про-гон,
    а не отдельно для каждого ролика.
    """
    videos = _discover_videos(data_dir)
    video_data = {v["vid"]: _batch_load_video(v) for v in videos}

    for group_name, group_cols, vram_est in _BATCH_GROUPS:
        if not any(group_cols - existing for _, existing in vid-eo_data.values()): сontinue
        for v in videos:
            acc, existing = video_data[v["vid"]]
            acc, existing, _ = _batch_run_group_for_video(group_name, v, acc, existing, config)
            video_data[v["vid"]] = (acc, existing)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    results = {}
    for v in videos:
        acc, _ = video_data[v["vid"]]
        results[v["vid"]] = _batch_finalize(acc, v["output_path"])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--retention-path", required=True)
    parser.add_argument("-v", "--video_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-c", "--config_path")
    parser.add_argument("-d", "--data_dir", default="data")
    args = parser.parse_args()
    config = Config(config_path=args.config_path)
    aggregated_df = aggregate(video_path=args.video_path, au-dio_path=args.video_path, output_path=args.output_path, config=config, re-tention_csv_path=args.retention_csv_path, data_dir=args.data_dir)
    aggregated_df.to_csv(args.output_path, index=True)