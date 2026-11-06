#!/usr/bin/env bash
set -e

source ./wandb_config.sh
load_wandb_config

# Use the regular trainer for plain, naive, and CSD-L2 experiments.
run_for_seeds() {
  for seed in 1; do
    python3 src/vit/train.py --seed "$seed" "$@"
  done
}

# CIFAR-100: plain Muon baseline
run_for_seeds --dataset "CIFAR100" --img-size 32 --num_channels 3 --num-classes 100 \
  --batch-size 256 --epochs 25 --d-model 192 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "muon" --lr 0.01 --momentum 0.9 --weight_decay 1e-5 \
  --save_dir ./checkpoints-vit/plain

# CIFAR-100: naive teleportation Muon (unchanged)
run_for_seeds --dataset "CIFAR100" --img-size 32 --num_channels 3 --num-classes 100 \
  --batch-size 256 --epochs 25 --d-model 192 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "muon" --lr 0.01 --momentum 0.9 --weight_decay 1e-5 \
  --tele-method naive --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --n-teleport 16 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1,5 \
  --save_dir ./checkpoints-vit/naive

# CIFAR-100: CSD-L2 Muon teleportation
run_for_seeds --dataset "CIFAR100" --img-size 32 --num_channels 3 --num-classes 100 \
  --batch-size 256 --epochs 25 --d-model 192 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "muon" --lr 0.01 --momentum 0.9 --weight_decay 1e-5 \
  --tele-method csd_l2 --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --tele-candidates 6 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1,11 \
  --save_dir ./checkpoints-vit/csd_l2_

# CIFAR-100: CSD-L2-Nesterov Muon teleportation
run_for_seeds --dataset "CIFAR100" --img-size 32 --num_channels 3 --num-classes 100 \
  --batch-size 256 --epochs 25 --d-model 192 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "muon" --lr 0.01 --momentum 0.9 --weight_decay 1e-5 \
  --tele-method csd_l2_nesterov --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --tele-candidates 6 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1,11 \
  --save_dir ./checkpoints-vit/csd_l2_nesterov
