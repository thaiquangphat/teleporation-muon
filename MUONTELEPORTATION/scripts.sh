#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/checkpoints/all_settings_seed${SEED}}"
DATASETS="${DATASETS:-MNIST CIFAR10}"

read -r -a DATASET_LIST <<< "$DATASETS"
STATE_MODES=(keep transport reset)
TELE_METHODS=(naive grad_norm muon_alignment one_step_loss)

mkdir -p "$RESULTS_DIR"
printf 'Running experiments: datasets=%s seed=%s epochs=%s batch_size=%s\n' \
  "$DATASETS" "$SEED" "$EPOCHS" "$BATCH_SIZE"
printf 'Results directory: %s\n' "$RESULTS_DIR"

for dataset in "${DATASET_LIST[@]}"; do
  dataset_args=(--dataset "$dataset")
  if [[ "$dataset" == "MNIST" ]]; then
    dataset_args+=(--img-size 28 --padding 4)
  elif [[ "$dataset" == "CIFAR10" ]]; then
    dataset_args+=(--img-size 32 --padding 4)
  else
    printf 'Unsupported dataset in DATASETS: %s\n' "$dataset" >&2
    exit 2
  fi

  common_args=(
    "${dataset_args[@]}"
    --seed "$SEED"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "${NUM_WORKERS:-0}"
    --opt muon
    --muon-lr "${MUON_LR:-0.02}"
    --muon-momentum "${MUON_MOMENTUM:-0.95}"
    --weight-decay "${WEIGHT_DECAY:-1e-4}"
    --hidden-dim "${HIDDEN_DIM:-128}"
    --save-dir "$RESULTS_DIR"
    --tele-candidates "${TELE_CANDIDATES:-6}"
    --tele-interval "${TELE_INTERVAL:-1}"
    --tele-low "${TELE_LOW:-0.7}"
    --tele-high "${TELE_HIGH:-1.5}"
    --tele-accept-tol "${TELE_ACCEPT_TOL:-1e-6}"
  )

  echo "=== $dataset: plain Muon baseline ==="
  "$PYTHON_BIN" src/train.py "${common_args[@]}" \
    --tele-method none --tele-state-mode keep

  for method in "${TELE_METHODS[@]}"; do
    for state_mode in "${STATE_MODES[@]}"; do
      echo "=== $dataset: method=$method state_mode=$state_mode ==="
      "$PYTHON_BIN" src/train.py "${common_args[@]}" \
        --tele-method "$method" --tele-state-mode "$state_mode"
    done
  done
done

echo "=== All training runs finished; plotting results ==="
"$PYTHON_BIN" plot_results.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$RESULTS_DIR/figures"

echo "Experiments and plots completed: $RESULTS_DIR"
