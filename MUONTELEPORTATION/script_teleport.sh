#MNIST Naive Teleport
python3 src/train.py --seed 1 --dataset "MNIST" --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 128 --epochs 20 --d-model 128 --intermediate-size 512 --num-heads 4 --position-embedding "rope" \
  --opt "sgd" --lr 0.015 --momentum 0.9 --weight_decay 1e-4 \
  --tele-method naive --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --n-teleport 16 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1 \
  --save_dir ./checkpoints/naive

#MNIST Naive Teleport Muon
python3 src/train.py --seed 1 --dataset "MNIST" --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 128 --epochs 20 --d-model 128 --intermediate-size 512 --num-heads 4 --position-embedding "rope" \
  --opt "muon" --lr 0.015 --momentum 0.9 --weight_decay 1e-4 \
  --tele-method naive --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --n-teleport 16 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1 \
  --save_dir ./checkpoints/naive

#MNIST Muon-alignment Teleport
python3 src/train.py --seed 1 --dataset "MNIST" --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 128 --epochs 20 --d-model 128 --intermediate-size 512 --num-heads 4 --position-embedding "rope" \
  --opt "muon" --lr 0.015 --momentum 0.9 --weight_decay 1e-4 \
  --tele-method muon_alignment --tele-layer "all" --tele-candidates 6 \
  --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1 \
  --tele-accept-tol 1e-6 --save_dir ./checkpoints/muon_alignment
