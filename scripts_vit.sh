
#!/usr/bin/env bash
set -e

source ./wandb_config.sh
load_wandb_config

# Use the regular trainer for all experiments.
run_for_seeds() {
    for seed in 1; do
        python3 src/vit/train.py --seed "$seed" "$@"
    done
}

# Common CIFAR-100 configuration.
COMMON_ARGS=(
    --dataset "CIFAR100"
    --img-size 32
    --num_channels 3
    --num-classes 100
    --batch-size 256
    --epochs 10
    --d-model 192
    --intermediate-size 768
    --num-heads 3
    --position-embedding "rope"
    --opt "muon"
    --lr 0.02
    --momentum 0.9
    --weight_decay 1e-5
)

# Common teleportation configuration.
TELE_ARGS=(
    --tele-att 1
    --tele-mlp 0
    --tele-opt 1
    --tele-sign 0
    --tele-layer "all"
    --tele-batch 4
    --tele-cons 4
    --tele-high 1.65
    --tele-low 0.35
    --tele-epoch-array 1
)

# CIFAR-100: plain Muon baseline (ordinary EMA).
run_for_seeds "${COMMON_ARGS[@]}" \
    --save_dir ./checkpoints-vit/plain

# CIFAR-100: naive teleportation Muon (ordinary EMA).
run_for_seeds "${COMMON_ARGS[@]}" "${TELE_ARGS[@]}" \
    --tele-method naive \
    --n-teleport 6 \
    --save_dir ./checkpoints-vit/naive

# CIFAR-100: CSD-L2 Muon teleportation (EMA scoring).
run_for_seeds "${COMMON_ARGS[@]}" "${TELE_ARGS[@]}" \
    --tele-method csd_l2 \
    --tele-candidates 6 \
    --save_dir ./checkpoints-vit/csd_l2

# CIFAR-100: CSD-L2-Nesterov with nuclear norm scoring.
run_for_seeds "${COMMON_ARGS[@]}" "${TELE_ARGS[@]}" \
    --tele-method csd_l2_nesterov \
    --nesterov \
    --tele-candidates 6 \
    --save_dir ./checkpoints-vit/csd_l2_nesterov

# CIFAR-100: CSD-L2-Nesterov with Frobenius norm scoring.
run_for_seeds "${COMMON_ARGS[@]}" "${TELE_ARGS[@]}" \
    --tele-method csd_l2_nesterov_frobenius \
    --nesterov \
    --tele-candidates 6 \
    --save_dir ./checkpoints-vit/csd_l2_nesterov_frobenius

# CIFAR-100: CSD-L2-Nesterov with post-orthogonalization nuclear norm scoring.
run_for_seeds "${COMMON_ARGS[@]}" "${TELE_ARGS[@]}" \
    --tele-method csd_l2_nesterov_post_orth \
    --nesterov \
    --tele-candidates 6 \
    --save_dir ./checkpoints-vit/csd_l2_nesterov_post_orth

# CIFAR-100: CSD-L2-Nesterov with post-orthogonalization Frobenius scoring.
run_for_seeds "${COMMON_ARGS[@]}" "${TELE_ARGS[@]}" \
    --tele-method csd_l2_nesterov_post_orth_frobenius \
    --nesterov \
    --tele-candidates 6 \
    --save_dir ./checkpoints-vit/csd_l2_nesterov_post_orth_frobenius
