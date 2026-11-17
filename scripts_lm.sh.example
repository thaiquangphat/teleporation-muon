#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the repository root:
#   bash scripts_lm.sh
#
# To run both paper positional-encoding variants:
#   POSITION_EMBEDDINGS="sinusoidal_ape rope" bash scripts_lm.sh
#
# This launcher follows the author's WikiText-103 model/data/teleport settings
# where they map cleanly to the step-based LM trainer. The optimizer remains
# Muon (with auxiliary AdamW) and the CSD criteria remain repository-specific.

source ./wandb_config.sh
load_wandb_config

# -----------------------------------------------------------------------------
# Data and hardware (author's WikiText setup: full WikiText-103, batch 96)
# -----------------------------------------------------------------------------
DATA_DIR="${WIKITEXT_DATA_DIR:-./data/wt103}"
HF_CACHE_DIR="${HF_CACHE_DIR:-./data/hf_cache}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-1111}"
DATA_SEED="${DATA_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-96}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
AMP_DTYPE="${AMP_DTYPE:-none}"

# The author's launcher includes both sinusoidal APE and RoPE experiments.
# Default to the WikiText paper setting (sinusoidal APE); set
# POSITION_EMBEDDINGS="sinusoidal_ape rope" to run both variants.
POSITION_EMBEDDINGS="${POSITION_EMBEDDINGS:-rope}"
read -r -a POSITION_EMBEDDING_VALUES <<< "${POSITION_EMBEDDINGS}"
if [[ ${#POSITION_EMBEDDING_VALUES[@]} -eq 0 ]]; then
  echo "POSITION_EMBEDDINGS must contain at least one value." >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Model (author's Transformer-XL dimensions, implemented in causal GPT-2 style)
# -----------------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-gpt2}" # tokenizer source; model weights are scratch initialized
D_MODEL="${D_MODEL:-128}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-16}"
NUM_HEADS="${NUM_HEADS:-8}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-256}"
ACTIVATION="${ACTIVATION:-relu}"
DROPOUT="${DROPOUT:-0.1}"
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
LAYER_NORM_EPSILON="${LAYER_NORM_EPSILON:-1e-5}"

# -----------------------------------------------------------------------------
# Step-based training
# Author values: max_step=500000, warmup_step=2000, cosine schedule.
# Logging/evaluation intervals are engineering choices, not paper parameters.
# -----------------------------------------------------------------------------
MAX_STEPS="${MAX_STEPS:-500000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_MIN="${LR_MIN:-0.0}"
LR_STEP_SIZE="${LR_STEP_SIZE:-1000}"
LR_GAMMA="${LR_GAMMA:-0.5}"

# -----------------------------------------------------------------------------
# Optimizer
# The author uses SGD at lr=0.75. That LR is not transferable to Muon.
# Keep the existing LM Muon/aux-Adam defaults and tune them independently.
# -----------------------------------------------------------------------------
MUON_LR="${MUON_LR:-0.002}"
ADAM_LR="${ADAM_LR:-5e-5}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-0}"

# -----------------------------------------------------------------------------
# Teleportation
# Author values: 8 candidates, alpha in [0.8, 1.2], 128 attempts arranged in
# groups of 16, between steps 500 and 1000, all attention layers, no MLP.
#
# This trainer uses 1-based optimizer steps and periodic step schedules.
# TELE_INTERVAL=500 plus START=501/LIMIT=1000 schedules exactly eight groups
# of 16 events in the second 500-step window (steps 501..1000). The first
# window is suppressed by START. This is the step-based analogue of the
# author's epoch/start/limit launcher; the trainer has no tele-epoch flag.
# -----------------------------------------------------------------------------
TELE_INTERVAL="${TELE_INTERVAL:-500}"
TELE_START_STEP="${TELE_START_STEP:-501}"
TELE_LIMIT_STEP="${TELE_LIMIT_STEP:-1000}"
TELE_BATCH="${TELE_BATCH:-128}"
TELE_CONS="${TELE_CONS:-16}"
N_TELEPORT="${N_TELEPORT:-8}"
TELE_CANDIDATES="${TELE_CANDIDATES:-8}"
TELE_HIGH="${TELE_HIGH:-1.2}"
TELE_LOW="${TELE_LOW:-0.8}"
TELE_SIGN="${TELE_SIGN:-0}"
TELE_ATT="${TELE_ATT:-1}"
TELE_MLP="${TELE_MLP:-0}"
TELE_LAYER="${TELE_LAYER:-all}"
TELE_OPT="${TELE_OPT:-1}"
CSD_SCORE_SCOPE="${CSD_SCORE_SCOPE:-all_muon}" # matches src/vit scoring scope
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints-lm}"

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --dataset-source local
  --dataset-config wikitext-103-raw-v1
  --cache-dir "$HF_CACHE_DIR"
  --model-name "$MODEL_NAME"
  --d-model "$D_MODEL"
  --num-hidden-layers "$NUM_HIDDEN_LAYERS"
  --num-heads "$NUM_HEADS"
  --intermediate-size "$INTERMEDIATE_SIZE"
  --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
  --activation "$ACTIVATION"
  --dropout "$DROPOUT"
  --attention-dropout "$ATTENTION_DROPOUT"
  --layer-norm-epsilon "$LAYER_NORM_EPSILON"
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
  local position_embedding="$1"
  local method="$2"
  shift 2

  local seed
  local -a seed_values
  IFS=',' read -r -a seed_values <<< "$SEEDS"
  for seed in "${seed_values[@]}"; do
    echo "============================================================"
    echo "PE=${position_embedding}; method=${method}; seed=${seed}"
    echo "steps=${MAX_STEPS}; warmup=${WARMUP_STEPS}; batch=${BATCH_SIZE}; fraction=${TRAIN_FRACTION}"
    echo "model=${D_MODEL}/${NUM_HIDDEN_LAYERS}/${NUM_HEADS}/${INTERMEDIATE_SIZE}; block=${BLOCK_SIZE}"
    echo "============================================================"
    python3 src/lm/train.py \
      "${COMMON_ARGS[@]}" \
      --position-embedding "$position_embedding" \
      --csd-score-scope "$CSD_SCORE_SCOPE" \
      --seed "$seed" \
      --tele-method "$method" \
      "$@"
  done
}

run_suite_for_position_embedding() {
  local position_embedding="$1"
  local pe_dir="$CHECKPOINT_ROOT/$position_embedding"

  # Plain Muon baseline.
  run_for_seeds "$position_embedding" none \
    --save-dir "$pe_dir/plain"

  # Naive teleportation: majority rule over global gradient L2.
  run_for_seeds "$position_embedding" naive \
    --n-teleport "$N_TELEPORT" \
    "${TELE_COMMON_ARGS[@]}" \
    --save-dir "$pe_dir/naive"

  # CSD-L2 on the ordinary EMA momentum buffer (no Nesterov flag).
  run_for_seeds "$position_embedding" csd_l2 \
    --tele-candidates "$TELE_CANDIDATES" \
    "${TELE_COMMON_ARGS[@]}" \
    --save-dir "$pe_dir/csd_l2"

  # Nesterov CSD criteria use the Nesterov-corrected direction.
  run_for_seeds "$position_embedding" csd_l2_nesterov \
    --nesterov --tele-candidates "$TELE_CANDIDATES" \
    "${TELE_COMMON_ARGS[@]}" \
    --save-dir "$pe_dir/csd_l2_nesterov"

  run_for_seeds "$position_embedding" csd_l2_nesterov_frobenius \
    --nesterov --tele-candidates "$TELE_CANDIDATES" \
    "${TELE_COMMON_ARGS[@]}" \
    --save-dir "$pe_dir/csd_l2_nesterov_frobenius"

  run_for_seeds "$position_embedding" csd_l2_nesterov_post_orth \
    --nesterov --tele-candidates "$TELE_CANDIDATES" \
    "${TELE_COMMON_ARGS[@]}" \
    --save-dir "$pe_dir/csd_l2_nesterov_post_orth"

  run_for_seeds "$position_embedding" csd_l2_nesterov_post_orth_frobenius \
    --nesterov --tele-candidates "$TELE_CANDIDATES" \
    "${TELE_COMMON_ARGS[@]}" \
    --save-dir "$pe_dir/csd_l2_nesterov_post_orth_frobenius"
}

for position_embedding in "${POSITION_EMBEDDING_VALUES[@]}"; do
  case "$position_embedding" in
    sinusoidal_ape|rope|ape|learned|learnable|learned_ape) ;;
    *)
      echo "Unsupported POSITION_EMBEDDINGS entry: $position_embedding" >&2
      exit 2
      ;;
  esac
  run_suite_for_position_embedding "$position_embedding"
done
