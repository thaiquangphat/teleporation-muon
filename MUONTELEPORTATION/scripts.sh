#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:-CIFAR10}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/checkpoints/${DATASET}_six_experiments_seed${SEED}}"
ADAMW_LR="${ADAMW_LR:-3e-4}"
MUON_LR="${MUON_LR:-0.015}"

case "$DATASET" in
  MNIST)
    DATA_ARGS=(--dataset MNIST --img-size 28)
    SGD_LR_DEFAULT="0.015"
    ;;
  CIFAR10)
    DATA_ARGS=(--dataset CIFAR10 --img-size 32)
    SGD_LR_DEFAULT="0.005"
    ;;
  *)
    printf 'Unsupported DATASET: %s (choose MNIST or CIFAR10)\n' "$DATASET" >&2
    exit 2
    ;;
esac
SGD_LR="${SGD_LR:-$SGD_LR_DEFAULT}"

COMMON_ARGS=(
  "${DATA_ARGS[@]}"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "${NUM_WORKERS:-0}"
  --d-model "${D_MODEL:-128}"
  --num-hidden-layer "${NUM_HIDDEN_LAYER:-4}"
  --num-heads "${NUM_HEADS:-4}"
  --intermediate-size "${INTERMEDIATE_SIZE:-512}"
  --patch-size "${PATCH_SIZE:-4}"
  --position-embedding "${POSITION_EMBEDDING:-rope}"
  --muon-lr "$MUON_LR"
  --muon-momentum "${MUON_MOMENTUM:-0.95}"
  --momentum "${MOMENTUM:-0.9}"
  --weight-decay "${WEIGHT_DECAY:-1e-4}"
  --save-dir "$RESULTS_DIR"
  --tele-candidates "${TELE_CANDIDATES:-6}"
  --tele-interval "${TELE_INTERVAL:-1}"
  --tele-start-step "${TELE_START_STEP:-0}"
  --tele-limit "${TELE_LIMIT:--1}"
  --tele-low "${TELE_LOW:-0.7}"
  --tele-high "${TELE_HIGH:-1.5}"
  --tele-accept-tol "${TELE_ACCEPT_TOL:-1e-6}"
)

mkdir -p "$RESULTS_DIR"
printf 'Running six ViT experiments: dataset=%s seed=%s epochs=%s batch_size=%s\n' \
  "$DATASET" "$SEED" "$EPOCHS" "$BATCH_SIZE"
printf 'Results directory: %s\n' "$RESULTS_DIR"

run_experiment() {
  local optimizer="$1"
  local method="$2"
  local learning_rate

  case "$optimizer" in
    muon) learning_rate="$MUON_LR" ;;
    adamw) learning_rate="$ADAMW_LR" ;;
    sgd) learning_rate="$SGD_LR" ;;
    *)
      printf 'Unsupported optimizer: %s\n' "$optimizer" >&2
      return 2
      ;;
  esac

  "$PYTHON_BIN" src/train.py "${COMMON_ARGS[@]}" \
    --opt "$optimizer" --lr "$learning_rate" \
    --tele-method "$method" --tele-state-mode reset
}

for optimizer in muon adamw sgd; do
  echo "=== $DATASET: baseline optimizer=$optimizer ==="
  run_experiment "$optimizer" none
done

for optimizer in muon adamw sgd; do
  echo "=== $DATASET: muon_alignment/reset optimizer=$optimizer ==="
  run_experiment "$optimizer" muon_alignment
done

echo "=== All six training runs finished; plotting results ==="
"$PYTHON_BIN" plot_results.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$RESULTS_DIR/figures"

echo "Experiments and plots completed: $RESULTS_DIR"
