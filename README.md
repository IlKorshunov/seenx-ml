# seenx-ml

## Структура

```text
.
├── main.py                    # CLI для сборки признаков, обучения и инференса
├── run_pipeline.sh            # полный локальный прогон пайплайна
├── run_all_experiments.sh     # seq/multimodal/VideoMAE/BERT эксперименты
├── run_loo_experiments.sh     # LOO-бенчмарк табличных моделей
├── configs/                   # конфиги локального запуска и весов моделей
├── data/                      # входные видео, retention.csv и снапшоты
├── output/                    # итоговые CSV с признаками
├── embeddings/                # кэш тяжелых CLIP/CLAP/VideoMAE/text embeddings
├── src/
│   ├── aggregator.py          # сборка всех признаков в один датасет
│   ├── extractors/
│   │   ├── video/             # визуальные признаки
│   │   ├── audio/             # аудио признаки
│   │   └── text/              # признаки из транскрипта и комментариев
│   ├── models/                # нейросетевые модели удержания
│   ├── analysis/              # отчеты, сравнения, кластеризация
│   ├── cutting_shots/         # поиск и нарезка бамперов
│   └── utils/                 # конфиги, кэш, выравнивание эмбеддингов
├── train/                     # обучение, LOO-эксперименты и служебные утилиты
├── tune_hp/                   # Optuna-тюнинг
└── tests/                     # тесты
```

## Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Для признаков спикера нужен эталонный портрет в `static/misc/speaker_face.png`. Основные локальные веса (`TransNetV2`, `ArcFace`, `YOLO-face`, `RAFT`) подтягиваются через `run_pipeline.sh`; модели HuggingFace, Whisper, CLIP, DeepFace и EasyOCR скачиваются при первом использовании.

## Запуск

Полный сценарий:

```bash
python download_data.py --count 5
./run_pipeline.sh
```

Отдельные шаги:

```bash
python main.py aggregate -v data/{video_id}/video.mp4 -o output/{video_id}_features.csv -c configs/local.json -r data/{video_id}/retention.csv
python main.py train --features_dir output --data_dir data --output_dir my_metrics --save_path static/weights/model.cbm
python -m train.transformer.train_transformer_seq_v1
python -m train.transformer.train_multimodal_seq
```

Эксперименты:

```bash
./run_loo_experiments.sh
./run_all_experiments.sh
RUN_OPTUNA=1 ./run_all_experiments.sh
```

## Какие признаки собираются

**Видео.** Яркость, резкость, визуальная энтропия и сложность кадра, цветовая температура и насыщенность, движение, zoom по optical flow, темп монтажа, короткие вставки, новизна сцены, скринкаст, оверлеи, лица, вероятность спикера, бамперы, плотность объектов. Границы сцен считаются через TransNetV2 или ensemble `TransNetV2 + CLAP + VideoMAE + RAFT`.

**Аудио.** Громкость, ZCR, spectral centroid и rolloff для полного микса, вокала и музыки после Demucs; speech/music/silence, фоновая музыка, prosody, pitch, паузы, beat sync, loudness dynamics, spectral flux, laughter, SFX и surprisal речи.

**Текст.** WPS, обращения к зрителю, слова-паразиты, сложность речи, культурные/NER-ссылки, hook score, вопросы, главы, intro/outro, рекламные сегменты, clickbait gap, curiosity gap, storytelling, viewer engagement, примеры, плотность информации, sentiment, topic sharpness и признаки комментариев.

**Мультимодальные признаки.** Синхронизация визуального и аудио ряда, ритм контента, narrative momentum, engagement surprise, MM embeddings и fusion эмоций в Ekman-признаки.

Часть старых или слишком дорогих признаков оставлена в коде для экспериментов, но удаляется из финального CSV как deprecated/redundant: например `aesthetic_score`, `depth_*`, `audio_*_similarity`, `visual_*_similarity`, сырые `sent_*` и голосовые эмоции после fusion.
