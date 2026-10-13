# Tiny CPU smoke: plain Muon
python3 src/train.py --seed 3 --dataset "MNIST" --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 16 --epochs 1 --d-model 32 --intermediate-size 64 --num-heads 4 --num-hidden-layer 1 \
  --patch-size 7 --position-embedding "rope" --opt "muon" --lr 0.015 --momentum 0.9 --weight_decay 1e-4 \
  --devices cuda --num-workers 0 --max-train-batches 2 --save_dir ./checkpoints/smoke_plain

# Tiny CPU smoke: naive teleport
python3 src/train.py --seed 3 --dataset "MNIST" --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 16 --epochs 1 --d-model 32 --intermediate-size 64 --num-heads 4 --num-hidden-layer 1 \
  --patch-size 7 --position-embedding "rope" --opt "muon" --lr 0.015 --momentum 0.9 --weight_decay 1e-4 \
  --tele-method naive --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --n-teleport 3 --tele-batch 1 --tele-cons 1 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1 \
  --devices cuda --num-workers 0 --max-train-batches 2 --save_dir ./checkpoints/smoke_naive

# Tiny CPU smoke: muon-alignment teleport
python3 src/train.py --seed 3 --dataset "MNIST" --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 16 --epochs 1 --d-model 32 --intermediate-size 64 --num-heads 4 --num-hidden-layer 1 \
  --patch-size 7 --position-embedding "rope" --opt "muon" --lr 0.015 --momentum 0.9 --weight_decay 1e-4 \
  --tele-method muon_alignment --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --tele-candidates 3 --tele-batch 1 --tele-cons 1 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1 \
  --tele-accept-tol 1e-6 --devices cuda --num-workers 0 --max-train-batches 2 \
  --save_dir ./checkpoints/smoke_muon_alignment
