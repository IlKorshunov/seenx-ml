#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EXPERIMENTS_ROOT="$ROOT_DIR/experiments/loo"
LOG_ROOT="$ROOT_DIR/experiments/_logs"
mkdir -p "$EXPERIMENTS_ROOT" "$LOG_ROOT"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_ROOT/loo_${RUN_TS}"
mkdir -p "$RUN_LOG_DIR"
MASTER_LOG="$RUN_LOG_DIR/all.log"

clean_stream() {
  perl -pe 's/\r/\n/g; s/\x1b\[[0-9;?]*[ -\/]*[@-~]//g; s/\x1b\][^\x07]*(?:\x07|\x1b\\)//g; s/\x1b[@-_]//g' \
    | grep -aEv '^[[:space:]]*$'
}

exec > >(tee >(clean_stream >> "$MASTER_LOG")) 2>&1

[[ -f "$ROOT_DIR/venv/bin/activate" ]] && source "$ROOT_DIR/venv/bin/activate"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-data}"
OUTPUT_DIR_FEATURES="${OUTPUT_DIR_FEATURES:-output}"
EMBEDDINGS_ROOT="${EMBEDDINGS_ROOT:-embeddings}"
CURVE_POINTS="${CURVE_POINTS:-20}"
RANDOM_SEED="${RANDOM_SEED:-42}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATIENCE="${PATIENCE:-30}"
DEVICE="${DEVICE:-cuda}"

echo "[loo] log=${RUN_LOG_DIR#$ROOT_DIR/} curve_points=$CURVE_POINTS seed=$RANDOM_SEED epochs=$EPOCHS"

FAILED=()

run_loo_arch() {
  local arch="$1"
  local out_dir="$EXPERIMENTS_ROOT/$arch"
  local log_file="$RUN_LOG_DIR/loo_${arch}.log"

  echo "[run] loo_$arch out=${out_dir#$ROOT_DIR/} log=${log_file#$ROOT_DIR/}"
  mkdir -p "$out_dir"
  local t_start=$SECONDS
  set +e
  "$PYTHON_BIN" train/transformer/train_multimodal_seq.py \
    --loo-all \
    --arch "$arch" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --output-dir-features "$OUTPUT_DIR_FEATURES" \
    --embeddings-root "$EMBEDDINGS_ROOT" \
    --output-dir "$out_dir" \
    --curve-points "$CURVE_POINTS" \
    --random-seed "$RANDOM_SEED" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --device "$DEVICE" \
    2>&1 | tee >(clean_stream > "$log_file")
  local rc=${PIPESTATUS[0]}
  set -e
  local elapsed=$(( SECONDS - t_start ))

  [[ $rc -ne 0 ]] && { echo "[fail] loo_$arch exit=$rc ${elapsed}s"; FAILED+=("$arch"); } || echo "[ok] loo_$arch ${elapsed}s"
  echo
}

run_loo_arch "lstm"
run_loo_arch "transformer"

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[done] failed=${#FAILED[@]}/2"
  printf '  - %s\n' "${FAILED[@]}"
  echo "log=${MASTER_LOG#$ROOT_DIR/}"
  exit 1
fi
echo "[done] all=2 log=${MASTER_LOG#$ROOT_DIR/}"
