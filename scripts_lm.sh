#!/usr/bin/env bash
set -Eeuo pipefail

source ./wandb_config.sh
load_wandb_config

# -----------------------------------------------------------------------------
# Data and hardware
# -----------------------------------------------------------------------------
DATA_DIR="${WIKITEXT_DATA_DIR:-./data/wt103}"
HF_CACHE_DIR="${HF_CACHE_DIR:-./data/hf_cache}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-1}"
DATA_SEED="${DATA_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.1}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
AMP_DTYPE="${AMP_DTYPE:-none}"

# -----------------------------------------------------------------------------
# Step-based training and compact W&B logging
# -----------------------------------------------------------------------------
MAX_STEPS="${MAX_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-$((MAX_STEPS / 20))}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_MIN="${LR_MIN:-1e-6}"
LR_STEP_SIZE="${LR_STEP_SIZE:-500}"
LR_GAMMA="${LR_GAMMA:-0.5}"

# -----------------------------------------------------------------------------
# Optimizer settings
# -----------------------------------------------------------------------------
MUON_LR="${MUON_LR:-0.002}"
ADAM_LR="${ADAM_LR:-5e-5}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-0}"

# -----------------------------------------------------------------------------
# Teleportation settings
# -----------------------------------------------------------------------------
TELE_INTERVAL="${TELE_INTERVAL:-0}"
TELE_START_STEP="${TELE_START_STEP:-1}"
TELE_LIMIT_STEP="${TELE_LIMIT_STEP:-0}"
TELE_BATCH="${TELE_BATCH:-4}"
TELE_CONS="${TELE_CONS:-4}"
N_TELEPORT="${N_TELEPORT:-6}"
TELE_CANDIDATES="${TELE_CANDIDATES:-6}"
TELE_HIGH="${TELE_HIGH:-1.65}"
TELE_LOW="${TELE_LOW:-0.35}"
TELE_SIGN="${TELE_SIGN:-0}"
TELE_ATT="${TELE_ATT:-1}"
TELE_MLP="${TELE_MLP:-0}"
TELE_LAYER="${TELE_LAYER:-all}"
TELE_OPT="${TELE_OPT:-1}"

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --dataset-source local
  --dataset-config wikitext-103-raw-v1
  --cache-dir "$HF_CACHE_DIR"
  --model-name gpt2
  --position-embedding rope
  --block-size "$BLOCK_SIZE"
  --train-fraction "$TRAIN_FRACTION"
  --data-seed "$DATA_SEED"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --max-steps "$MAX_STEPS"
  --warmup-steps "$WARMUP_STEPS"
  --log-interval "$LOG_INTERVAL"
  --eval-interval "$EVAL_INTERVAL"
  --devices "$DEVICE"
  --amp-dtype "$AMP_DTYPE"
  --max-train-batches "$MAX_TRAIN_BATCHES"
  --max-eval-batches "$MAX_EVAL_BATCHES"
  --opt muon
  --lr "$MUON_LR"
  --adam-lr "$ADAM_LR"
  --momentum "$MOMENTUM"
  --weight-decay "$WEIGHT_DECAY"
  --grad-clip "$GRAD_CLIP"
  --lr-scheduler "$LR_SCHEDULER"
  --lr-min "$LR_MIN"
  --lr-step-size "$LR_STEP_SIZE"
  --lr-gamma "$LR_GAMMA"
)

TELE_COMMON_ARGS=(
  --tele-att "$TELE_ATT"
  --tele-mlp "$TELE_MLP"
  --tele-layer "$TELE_LAYER"
  --tele-high "$TELE_HIGH"
  --tele-low "$TELE_LOW"
  --tele-sign "$TELE_SIGN"
  --tele-batch "$TELE_BATCH"
  --tele-cons "$TELE_CONS"
  --tele-opt "$TELE_OPT"
  --tele-interval "$TELE_INTERVAL"
  --tele-start-step "$TELE_START_STEP"
  --tele-limit-step "$TELE_LIMIT_STEP"
)

run_for_seeds() {
  local method="$1"
  shift
  local seed
  IFS=',' read -r -a seed_values <<< "$SEEDS"
  for seed in "${seed_values[@]}"; do
    echo "============================================================"
    echo "Running method=${method} seed=${seed}; max_steps=${MAX_STEPS}; train_fraction=${TRAIN_FRACTION}"
    echo "============================================================"
    python3 src/lm/train.py \
      "${COMMON_ARGS[@]}" \
      --seed "$seed" \
      --tele-method "$method" \
      "$@"
  done
}

# Plain Muon baseline.
run_for_seeds none \
  --save-dir ./checkpoints-lm/plain

# Naive teleportation: majority rule over global gradient L2.
run_for_seeds naive \
  --n-teleport "$N_TELEPORT" \
  "${TELE_COMMON_ARGS[@]}" \
  --save-dir ./checkpoints-lm/naive

# CSD-L2 on the ordinary EMA momentum buffer (no Nesterov flag).
run_for_seeds csd_l2 \
  --tele-candidates "$TELE_CANDIDATES" \
  "${TELE_COMMON_ARGS[@]}" \
  --save-dir ./checkpoints-lm/csd_l2

# Nesterov CSD criteria all use the Nesterov-corrected direction.
run_for_seeds csd_l2_nesterov \
  --nesterov --tele-candidates "$TELE_CANDIDATES" \
  "${TELE_COMMON_ARGS[@]}" \
  --save-dir ./checkpoints-lm/csd_l2_nesterov

run_for_seeds csd_l2_nesterov_frobenius \
  --nesterov --tele-candidates "$TELE_CANDIDATES" \
  "${TELE_COMMON_ARGS[@]}" \
  --save-dir ./checkpoints-lm/csd_l2_nesterov_frobenius

run_for_seeds csd_l2_nesterov_post_orth \
  --nesterov --tele-candidates "$TELE_CANDIDATES" \
  "${TELE_COMMON_ARGS[@]}" \
  --save-dir ./checkpoints-lm/csd_l2_nesterov_post_orth

run_for_seeds csd_l2_nesterov_post_orth_frobenius \
  --nesterov --tele-candidates "$TELE_CANDIDATES" \
  "${TELE_COMMON_ARGS[@]}" \
  --save-dir ./checkpoints-lm/csd_l2_nesterov_post_orth_frobenius
