#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-42 123 2024}"
EPOCHS="${EPOCHS:-5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_DIR="${SAVE_DIR:-./checkpoints}"
read -r -a seed_array <<< "$SEEDS"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi

for dataset in MNIST CIFAR10; do
    for optimizer in adamw sgd muon; do
        for mode in baseline teleport; do
            if [[ "$mode" == "baseline" ]]; then
                teleport_count=0
            else
                teleport_count=8
            fi

            for seed in "${seed_array[@]}"; do
                echo "Running dataset=$dataset optimizer=$optimizer mode=$mode seed=$seed"
                "$PYTHON" ./src/train.py \
                    --dataset "$dataset" \
                    --epochs "$EPOCHS" \
                    --batch-size "$BATCH_SIZE" \
                    --num-workers "$NUM_WORKERS" \
                    --lr 1e-3 \
                    --muon-lr 0.02 \
                    --opt "$optimizer" \
                    --seed "$seed" \
                    --save-dir "$SAVE_DIR" \
                    --n-teleport "$teleport_count" \
                    --tele-batch 32 \
                    --tele-opt 1 \
                    --tele-high 1.1 \
                    --tele-low 0.9 \
                    --tele-cons 16 \
                    --tele-layer all
            done
        done
    done
done

echo "All experiments completed."
