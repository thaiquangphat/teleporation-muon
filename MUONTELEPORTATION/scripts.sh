set -e

run_for_seeds() {
  for seed in 1; do
    python3 src/train.py --seed "$seed" "$@"
  done
}

#CIFAR-100 No teleport Muon
run_for_seeds --dataset "CIFAR100" --img-size 32 --num_channels 3 --num-classes 10 \
  --batch-size 256 --epochs 50 --d-model 192 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "muon" --lr 0.005 --momentum 0.9 --weight_decay 1e-5 --save_dir ./checkpoints/plain

#CIFAR-100 Naive Teleport Muon
run_for_seeds --dataset "CIFAR100" --img-size 32 --num_channels 3 --num-classes 10 \
  --batch-size 256 --epochs 50 --d-model 192 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "muon" --lr 0.005 --momentum 0.9 --weight_decay 1e-5 \
  --tele-method naive --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --n-teleport 16 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1 \
  --save_dir ./checkpoints/naive

#CIFAR-100 Muon-alignment Teleport
run_for_seeds --dataset "CIFAR100" --img-size 32 --num_channels 3 --num-classes 10 \
  --batch-size 256 --epochs 50 --d-model 192 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "muon" --lr 0.005 --momentum 0.9 --weight_decay 1e-5 \
  --tele-method muon_alignment --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --tele-candidates 6 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1 \
  --tele-accept-tol 1e-6 --save_dir ./checkpoints/muon_alignment

python3 plot_results.py --results-dir ./checkpoints --output-dir ./plots
