#!/usr/bin/env bash
set -Eeuo pipefail

source ./wandb_config.sh
load_wandb_config

DATA_DIR="${WIKITEXT_DATA_DIR:-./data/wt103}"
HF_CACHE_DIR="${HF_CACHE_DIR:-./data/hf_cache}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
EPOCHS="${EPOCHS:-3}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
AMP_DTYPE="${AMP_DTYPE:-none}"
TELE_EPOCHS="${TELE_EPOCHS:-1}"
TELE_BATCH="${TELE_BATCH:-4}"
TELE_CONS="${TELE_CONS:-4}"
N_TELEPORT="${N_TELEPORT:-6}"
TELE_CANDIDATES="${TELE_CANDIDATES:-6}"

run_for_seeds() {
  local seed
  IFS=',' read -r -a seed_values <<< "$SEEDS"
  for seed in "${seed_values[@]}"; do
    python3 src/lm/train.py \
      --seed "$seed" \
      --data-dir "$DATA_DIR" \
      --dataset-source local \
      --dataset-config wikitext-103-raw-v1 \
      --cache-dir "$HF_CACHE_DIR" \
      --model-name gpt2 \
      --position-embedding rope \
      --block-size "$BLOCK_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --eval-batch-size "$EVAL_BATCH_SIZE" \
      --num-workers 2 \
      --epochs "$EPOCHS" \
      --devices "$DEVICE" \
      --amp-dtype "$AMP_DTYPE" \
      --max-train-batches "$MAX_TRAIN_BATCHES" \
      --max-eval-batches "$MAX_EVAL_BATCHES" \
      --opt muon \
      --lr 0.002 \
      --adam-lr 5e-5 \
      --momentum 0.9 \
      --weight-decay 0.01 \
      --lr-scheduler cosine \
      "$@"
  done
}

# Plain Muon baseline.
run_for_seeds \
  --tele-method none \
  --nesterov False \
  --save_dir ./checkpoints-lm/plain

# ViT-style naive teleportation: gradient-L2 majority rule, followed by
# ordinary Muon momentum (Nesterov is intentionally disabled). Existing
# optimizer state is transported through the committed GPT-2 symmetry.
run_for_seeds \
  --tele-method naive \
  --nesterov False \
  --tele-att 1 --tele-mlp 0 --tele-layer all \
  --n-teleport "$N_TELEPORT" --tele-high 1.65 --tele-low 0.35 --tele-sign 0 \
  --tele-epoch-array "$TELE_EPOCHS" \
  --tele-batch "$TELE_BATCH" --tele-cons "$TELE_CONS" --tele-opt 1 \
  --save_dir ./checkpoints-lm/naive

# Current framework: CSD-L2 and the actual Muon update both use Nesterov.
# CSD is computed on each teleported candidate's Nesterov direction;
# identity wins exact ties.
run_for_seeds \
  --nesterov \
  --nesterov True \
  --tele-method csd_l2_nesterov \
  --tele-att 1 --tele-mlp 0 --tele-layer all \
  --tele-candidates "$TELE_CANDIDATES" --tele-high 1.65 --tele-low 0.35 --tele-sign 0 \
  --tele-epoch-array "$TELE_EPOCHS" \
  --tele-batch "$TELE_BATCH" --tele-cons "$TELE_CONS" --tele-opt 1 \
  --save_dir ./checkpoints-lm/csd_l2_nesterov
