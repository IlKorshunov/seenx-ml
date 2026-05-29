#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-venv}"
SKIP_WEIGHTS="${SKIP_WEIGHTS:-0}"

echo "[setup] root=$ROOT_DIR"
echo "[setup] python=$PYTHON_BIN"
echo "[setup] venv=$VENV_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] creating venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[setup] upgrade pip tools"
python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing dependecies"
python -m pip install -r requirements.txt

echo "[setup] prepare folders"
mkdir -p static/weights output embeddings data my_metrics experiments src/tune_hp/results src/get_data/comments

download_if_missing() {
  local path="$1"
  local url="$2"
  if [[ -f "$path" ]]; then
    echo "[weights] exists: $path"
    return
  fi

  echo "[weights] download: $(basename "$path")"
  mkdir -p "$(dirname "$path")"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$path" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$path" "$url"
  else
    echo "[warn] neither curl nor wget is available; cannot download $path" >&2
    return 0
  fi
}

if [[ "$SKIP_WEIGHTS" != "1" ]]; then
  echo "[setup] download local weights"
  download_if_missing "static/weights/transnetv2-pytorch-weights.pth" \
    "https://huggingface.co/Sn4kehead/TransNetV2/resolve/main/transnetv2-pytorch-weights.pth?download=true"
  download_if_missing "static/weights/arcface_weights.h5" \
    "https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5"
  download_if_missing "static/weights/yolov12l-face.pt" \
    "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12l-face.pt"
  download_if_missing "static/weights/raft-small.pth" \
    "https://huggingface.co/ddrfan/RAFT/resolve/main/raft-small.pth?download=true"

  if [[ ! -d "RAFT" ]]; then
    if command -v git >/dev/null 2>&1; then
      echo "[setup] cloning RAFT"
      git clone --depth 1 https://github.com/princeton-vl/RAFT.git RAFT
    else
      echo "[warn] git is not available; skip RAFT clone" >&2
    fi
  else
    echo "[weights] exists: RAFT"
  fi
fi

echo "[setup] install optional spacy ru model"
python -m spacy download ru_core_news_sm || {
  echo "[warn] failed to install ru_core_news_sm automatically" >&2
  echo "you can install it later: source $VENV_DIR/bin/activate && python -m spacy download ru_core_news_sm" >&2
}

echo "[setup] check imports"
python - <<'PY'
modules = [
    "numpy",
    "pandas",
    "sklearn",
    "torch",
    "transformers",
    "cv2",
    "catboost",
    "optuna",
    "googleapiclient",
    "pytest",
]
missing = []
for name in modules:
    try:
        __import__(name)
    except Exception as exc:
        missing.append((name, str(exc)))
if missing:
    print("[warn] some imports failed:")
    for name, err in missing:
        print(f"  - {name}: {err}")
else:
    print("[setup] core imports OK")
PY

cat <<EOF
Activate environment:
  source "$VENV_DIR/bin/activate"

Run tests:
  pytest

Run pipeline:
  ./run_pipeline.sh
EOF
