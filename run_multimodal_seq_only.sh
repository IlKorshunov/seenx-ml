#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT_DIR"

PATIENCE="${PATIENCE:-50}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --patience) PATIENCE="${2:-50}"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

EXPERIMENTS_ROOT="$ROOT_DIR/experiments"; LOG_ROOT="$ROOT_DIR/experiments/_logs"
mkdir -p "$EXPERIMENTS_ROOT" "$LOG_ROOT"
RUN_TS="$(date +%Y%m%d_%H%M%S)"; RUN_LOG_DIR="$LOG_ROOT/multimodal_seq_${RUN_TS}"
mkdir -p "$RUN_LOG_DIR"; MASTER_LOG="$RUN_LOG_DIR/all.log"

clean_stream() {
  perl -pe 's/\r/\n/g; s/\x1b\[[0-9;?]*[ -\/]*[@-~]//g; s/\x1b\][^\x07]*(?:\x07|\x1b\\)//g; s/\x1b[@-_]//g' \
    | grep -aEv '^(Training:|Epoch [0-9]+ \[(train|val)\]:|Ep [0-9]+ \[(train|val)\]:)' \
    | grep -aEv '(^[[:space:]]*[0-9]+%?\|)|(\|[[:space:]]*[0-9]+/[0-9]+[[:space:]]*\[)|(\?[[:space:]]*[A-Za-z]+/s\]$)' \
    | grep -aEv '^[[:space:]]*$'
}

exec > >(tee >(clean_stream >> "$MASTER_LOG")) 2>&1

[[ -f "$ROOT_DIR/seenx-ml-venv/bin/activate" ]] && source "$ROOT_DIR/seenx-ml-venv/bin/activate"
[[ -f "$ROOT_DIR/venv/bin/activate" ]] && source "$ROOT_DIR/venv/bin/activate"

PYTHON_BIN="${PYTHON_BIN:-python3}"; DEVICE="${DEVICE:-cuda}"; VAL_N="${VAL_N:-10}"; EPOCHS="${EPOCHS:-1000}"
HEAVY_BATCH="${HEAVY_BATCH:-16}"; HEAVY_D_MODEL="${HEAVY_D_MODEL:-512}"
HEAVY_N_LAYERS_LSTM="${HEAVY_N_LAYERS_LSTM:-4}"; HEAVY_N_LAYERS_TRF="${HEAVY_N_LAYERS_TRF:-6}"
HEAVY_N_HEADS="${HEAVY_N_HEADS:-8}"; HEAVY_D_FF="${HEAVY_D_FF:-1024}"

CURVE_POINTS="${CURVE_POINTS:-0}"; TIME_FEATURES="${TIME_FEATURES:-none}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-540}"; MAX_DURATION_SEC="${MAX_DURATION_SEC:-1620}"
SEQ_EXTRA_ARGS=()
[[ "$CURVE_POINTS" != "0" ]] && SEQ_EXTRA_ARGS+=(--curve-points "$CURVE_POINTS")
[[ "$TIME_FEATURES" != "none" ]] && SEQ_EXTRA_ARGS+=(--time-features "$TIME_FEATURES")
[[ "$MIN_DURATION_SEC" != "0" ]] && SEQ_EXTRA_ARGS+=(--min-duration-sec "$MIN_DURATION_SEC")
[[ "$MAX_DURATION_SEC" != "0" ]] && SEQ_EXTRA_ARGS+=(--max-duration-sec "$MAX_DURATION_SEC")

TUNE_LSTM_JSON="${TUNE_LSTM_JSON:-$ROOT_DIR/tune_hp/results/tune_multimodal_lstm_best.json}"
TUNE_TRF_JSON="${TUNE_TRF_JSON:-$ROOT_DIR/tune_hp/results/tune_multimodal_transformer_best.json}"

run_exp() {
  local name="$1" out_dir="$2"; shift 2; local cmd=("$@")
  local abs_out="$ROOT_DIR/$out_dir" log_file="$RUN_LOG_DIR/${name}.log"
  [[ -d "$abs_out" ]] && rm -rf "$abs_out"; mkdir -p "$abs_out"
  set +e; "${cmd[@]}" 2>&1 | tee >(clean_stream > "$log_file"); local rc=${PIPESTATUS[0]}; set -e
  [[ $rc -ne 0 ]] && FAILED_EXPERIMENTS+=("$name")
}

tuned_args() { local json="$1"; [[ -f "$json" ]] && printf '%s\n%s\n' --tuned-params-json "$json"; }

FAILED_EXPERIMENTS=()

if [[ "${RUN_OPTUNA:-0}" == "1" ]]; then
  N_OPTUNA="${N_OPTUNA_TRIALS:-5}"; O_EPOCHS="${OPTUNA_EPOCHS_PER_TRIAL:-150}"
  mkdir -p "$ROOT_DIR/tune_hp/results"
  for arch in multimodal_transformer multimodal_lstm; do
    set +e
    "$PYTHON_BIN" "$ROOT_DIR/tune_hp/tune.py" --arch "$arch" --n-trials "$N_OPTUNA" \
      --epochs-per-trial "$O_EPOCHS" --device "$DEVICE" \
      --output-dir "$ROOT_DIR/tune_hp/results" --val-first-n-output "$VAL_N"
    set -e
    study_json="tune_${arch}_best.json"
    [[ -f "$ROOT_DIR/tune_hp/results/$study_json" ]] && cp -f "$ROOT_DIR/tune_hp/results/$study_json" "$ROOT_DIR/$study_json"
  done
fi

if [[ "${SKIP_CLUSTER_JSON:-0}" != "1" ]]; then
  "$PYTHON_BIN" -m train.clustering.cluster_specialists_multimodal --repo-root "$ROOT_DIR" \
    --features-root "$ROOT_DIR/data" --clusters-json "$ROOT_DIR/configs/video_clusters.json" \
    --lists-json "$ROOT_DIR/configs/video_cluster_train_lists.json" || true
fi

if [[ "${SKIP_NEW_CLUSTERING:-0}" != "1" ]]; then
  "$PYTHON_BIN" analysis/video_clustering.py --min-k 6 --max-k 8 --strategy retention || true
  "$PYTHON_BIN" src/analysis/cluster_and_curve_features.py \
    --clusters-file analysis/video_clustering/retention/clusters.json || true
fi

mapfile -t TUNE_LSTM_ARGS < <(tuned_args "$TUNE_LSTM_JSON")

run_exp "lstm_v3_multimodal" "experiments/lstm_exp/v3_multimodal" \
  "$PYTHON_BIN" train/train_multimodal_seq.py --arch lstm \
  --output-dir "experiments/lstm_exp/v3_multimodal" --output-dir-features output \
  --snapshot-dir data --embeddings-root embeddings --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" --n-layers "$HEAVY_N_LAYERS_LSTM" \
  --epochs "$EPOCHS" --batch-size "$HEAVY_BATCH" "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" --device "$DEVICE"

run_exp "lstm_v4_tuned_multimodal" "experiments/lstm_exp/v4_tuned_multimodal" \
  "$PYTHON_BIN" train/train_multimodal_seq.py --arch lstm \
  --output-dir "experiments/lstm_exp/v4_tuned_multimodal" --output-dir-features output \
  --snapshot-dir data --embeddings-root embeddings --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" --n-layers "$HEAVY_N_LAYERS_LSTM" "${TUNE_LSTM_ARGS[@]}" \
  --epochs "$EPOCHS" --batch-size "$HEAVY_BATCH" "${SEQ_EXTRA_ARGS[@]}" \
  --patience "$PATIENCE" --device "$DEVICE"

mapfile -t TUNE_TRF_ARGS < <(tuned_args "$TUNE_TRF_JSON")

run_exp "transformer_v3_multimodal" "experiments/transformer_exp/v3_multimodal" \
  "$PYTHON_BIN" train/train_multimodal_seq.py --arch transformer \
  --output-dir "experiments/transformer_exp/v3_multimodal" --output-dir-features output \
  --snapshot-dir data --embeddings-root embeddings --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" --n-heads "$HEAVY_N_HEADS" --n-layers "$HEAVY_N_LAYERS_TRF" \
  --d-ff "$HEAVY_D_FF" --epochs "$EPOCHS" --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" --patience "$PATIENCE" --device "$DEVICE"

run_exp "transformer_v4_tuned_multimodal" "experiments/transformer_exp/v4_tuned_multimodal" \
  "$PYTHON_BIN" train/train_multimodal_seq.py --arch transformer \
  --output-dir "experiments/transformer_exp/v4_tuned_multimodal" --output-dir-features output \
  --snapshot-dir data --embeddings-root embeddings --val-first-n-output "$VAL_N" \
  --d-model "$HEAVY_D_MODEL" --n-heads "$HEAVY_N_HEADS" --n-layers "$HEAVY_N_LAYERS_TRF" \
  --d-ff "$HEAVY_D_FF" "${TUNE_TRF_ARGS[@]}" --epochs "$EPOCHS" --batch-size "$HEAVY_BATCH" \
  "${SEQ_EXTRA_ARGS[@]}" --patience "$PATIENCE" --device "$DEVICE"

"$PYTHON_BIN" "$ROOT_DIR/train/summarize_experiments.py" --root "$ROOT_DIR"

[[ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]] && { printf '  - %s\n' "${FAILED_EXPERIMENTS[@]}"; exit 1; }
exit 0