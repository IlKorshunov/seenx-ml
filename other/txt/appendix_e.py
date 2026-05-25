logger = logging.getLogger(__name__)

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    a = parser.add_argument
    a("--output-dir-features", default="output")
    a("--snapshot-dir", default="data")
    a("--output-dir", default="")
    a("--val-ratio", type=float, default=0.15)
    a("--val-first-n-output", type=int, default=0)
    a("--loo-all", action="store_true", default=False)
    a("--window-size", type=int, default=128)
    a("--d-model", type=int, default=128)
    a("--n-layers", type=int, default=4)
    a("--epochs", type=int, default=200)
    a("--batch-size", type=int, default=16)
    a("--lr", type=float, default=5e-4)
    a("--patience", type=int, default=30)
    # аналогично остальные параметры архитектуры,
    return parser


def init_run(args: argparse.Namespace) -> torch.device:
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    return torch.device(args.device)


def load_and_filter_data(args):
    """Загрузка посекундных табличных представлений всех видеороликов
    и фильтрация признакового пространства"""
    video_dfs = load_all_merged(args.output_dir_features, args.snapshot_dir, use_curve_raw=args.use_curve_raw)
    video_ids = sorted(video_dfs.keys())
    output_video_ids = resolve_output_video_ids(args.output_dir_features, video_dfs)
    feature_cols, _ = filter_features(video_dfs, **build_tuned_feature_filter_kwargs(args))
    return video_dfs, video_ids, output_video_ids, feature_cols


def resolve_split(args, video_ids, output_video_ids):
    """Разбиение выборки на обучающую и валидационную части по схеме"""
    train_ids, val_ids = resolve_train_val_split(args, video_ids, output_video_ids)
    return apply_train_id_file_filter(train_ids, args), val_ids


def make_normalizer(args, video_dfs, train_ids, feature_cols):
    """Вычисление нормализующих статистик на обучающей выборке."""
    normalizer = FeatureNormalizer()
    normalizer.fit({v: video_dfs[v] for v in train_ids}, feature_cols)
    weights = load_video_weights(train_ids, args.snapshot_dir) if args.engagement_weight else None
    return normalizer, max_time_sec_over_videos(video_dfs, train_ids), weights


def compute_baseline_curve(video_dfs, train_ids, normalizer):
    """Усреднённая опорная кривая удержания канала"""
    max_len = max(len(video_dfs[v]) for v in train_ids)
    acc, cnt = np.zeros(max_len), np.zeros(max_len)
    for v in train_ids:
        ret = pd.to_numeric(video_dfs[v]["retention"], errors="coerce").fillna(0).values
        acc[: len(ret)] += ret
        cnt[: len(ret)] += 1.0
    raw = (acc / np.maximum(cnt, 1.0)).astype(np.float32)
    return raw, normalizer.normalize_retention(raw).astype(np.float32)


def apply_global_calibration(train_true_list, train_pred_list):
    """Опциональная калибровка предсказаний"""
    cal_true = np.concatenate(train_true_list)
    cal_pred = np.concatenate(train_pred_list)
    A = np.vstack([cal_pred, np.ones(len(cal_pred))]).T
    cal_a, _ = np.linalg.lstsq(A, cal_true, rcond=None)[0:2]
    cal_a = float(np.clip(cal_a, 0.5, 2.0))
    cal_b = float(np.clip(np.mean(cal_true) - cal_a * np.mean(cal_pred), -50.0, 50.0))
    return cal_a, cal_b


def predict_all_videos(*, video_ids, val_ids, video_dfs, predict_fn,
                       output_dir, calibration=(1.0, 0.0)):
    """Инференс по всем роликам выборки и вычисление метрик качества."""
    cal_a, cal_b = calibration
    all_metrics = {}
    for vid in video_ids:
        split = "val" if vid in val_ids else "train"
        y_true, y_pred = predict_fn(vid)
        y_pred_cal = (cal_a * y_pred + cal_b).astype(y_pred.dtype)
        metrics = seq_metrics(y_pred_cal, y_true)
        all_metrics[vid] = {**metrics, "split": split, "n_seconds": len(y_true)}
        # сохранение графика предсказания для каждого ролика
    return {"all_metrics": all_metrics}


def run_loo_all(args, module_name: str) -> bool:
    """Запуск опционального loo режима"""
    if not args.loo_all:
        return False
    ids = sorted(p.name.replace("_features.csv", "")
                 for p in Path(args.output_dir_features).glob("*_features.csv"))
    base = Path(args.output_dir or "loo_runs")
    base.mkdir(parents=True, exist_ok=True)
    for idx, vid in enumerate(ids):
        out_dir = base / f"loo_{idx:03d}_{vid}"
        subprocess.run([sys.executable, "-m", module_name, *sys.argv[1:],
                        "--eval-video", vid, "--output-dir", str(out_dir)],
                       check=True)
        # агрегация метрик из out_dir/metrics.json в общую таблицу
    return True


def save_metrics_json(args, *, model_name, feature_cols, n_feat,
                      train_ids, val_ids, result, all_metrics,
                      feature_importance_meta=None):
    """Сохранение полного состояния эксперимента"""
    payload = {
        "model": model_name,
        "n_features": n_feat,
        "feature_cols": feature_cols,
        "train_ids": train_ids, "val_ids": val_ids,
        "best_val_loss": result.get("best_val_loss"),
        "epochs_trained": result["epochs_trained"],
        "per_video": all_metrics,
        "config": {k: v for k, v in vars(args).items()
                   if isinstance(v, (int, float, str, bool))},
    }
    if feature_importance_meta:
        payload["feature_importance"] = feature_importance_meta
    Path(os.path.join(args.output_dir, "metrics.json")).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _run_optuna_and_apply(args):
    """Запуск этапа подбора гиперпараметров."""
    study_name = f"tune_multimodal_{args.arch}"
    subprocess.run([sys.executable, str(TUNE_SCRIPT),
                    "--arch", f"multimodal_{args.arch}",
                    "--n-trials", str(args.n_trials),
                    "--study-name", study_name,
                    # оставшиеся аргументы
                    ], check=True)
    best = json.loads((Path(args.tune_output_dir) / f"{study_name}_best.json").read_text())
    apply_best_params_to_args(args, best.get("best_params", {}),
                              model_family=f"multimodal_{args.arch}",
                              apply_architecture=True)
    return args


def main():
    args = parse_args()
    # Опциональный режим
    if run_loo_all(args, "train.transformer.train_multimodal_seq"):
        return
    device = init_run(args)

    # Опциональный подбор гиперпараметров
    if args.tune_first:
        args = _run_optuna_and_apply(args)
    elif args.tuned_params_json:
        apply_params(args, f"multimodal_{args.arch}")

    # Подготовка данных: загрузка, фильтрация, разбиение, нормализация
    video_dfs, video_ids, output_video_ids, feature_cols = load_and_filter_data(args)
    video_embeddings = load_aligned_embeddings_for_videos(video_dfs, args.embeddings_root)
    train_ids, val_ids = resolve_split(args, video_ids, output_video_ids)
    normalizer, ref_sec, video_weights = make_normalizer(args, video_dfs, train_ids, feature_cols)

    # Построение датасетов  и инициализация модели
    train_ds = MultimodalWindowedDataset(video_dfs, video_embeddings, train_ids,
                                          feature_cols, normalizer,
                                          args.window_size, args.window_stride,
                                          video_weights=video_weights)
    val_ds = MultimodalWindowedDataset(video_dfs, video_embeddings, val_ids,
                                        feature_cols, normalizer,
                                        args.window_size, args.window_stride)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    n_tab = len(feature_cols)
    model_cls = MultimodalRetentionLSTM if args.arch == "lstm" else MultimodalRetentionTransformer
    model = model_cls(..., n_tabular_features=n_tab).to(device)
    set_model_baseline(model, video_dfs, train_ids, normalizer)
    # Обучение модели
    model, result = _train(model, train_dl, val_dl, device, args, args.engagement_weight)

    # Опциональная калибровка предсказаний
    if args.global_calibration:
        train_true, train_pred = zip(*[_predict_fn_raw(vid) for vid in train_ids])
        cal_a, cal_b = apply_global_calibration(list(train_true), list(train_pred))
    else:
        cal_a, cal_b = 1.0, 0.0

    # Инференс и оценка
    pred_out = predict_all_videos(video_ids=video_ids, val_ids=val_ids,
                                   video_dfs=video_dfs,
                                   predict_fn=_predict_fn_raw,
                                   output_dir=args.output_dir,
                                   calibration=(cal_a, cal_b))

    # Анализ значимости признаков и аблация модальностей
    fi_meta = _run_feature_importance(model, feature_cols, video_dfs,
                                       video_embeddings, val_ids, video_ids,
                                       normalizer, device, args,
                                       ref_sec, cal_a, cal_b)
    fi_meta.update(run_video_clustering_if_requested(args))

    # Сохранение полного состояния эксперимента
    save_metrics_json(args, model_name=f"Multimodal{args.arch.title()}",
                      feature_cols=feature_cols, n_feat=n_tab,
                      train_ids=train_ids, val_ids=val_ids,
                      result=result, all_metrics=pred_out["all_metrics"],
                      feature_importance_meta=fi_meta)
    torch.save({"model_state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "normalizer_median": normalizer.median.tolist(),
                "normalizer_iqr": normalizer.iqr.tolist(),
                # оставшиеся параметры
                }, os.path.join(args.output_dir, f"multimodal_{args.arch}_model.pt"))
