#MNIST Teleport
python3 src/train.py --seed 3 --dataset "MNIST" --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 128 --epochs 20 --d-model 128 --intermediate-size 512 --num-heads 4 --position-embedding "rope" \
  --opt "sgd" --lr 0.015 --momentum 0.9 --weight_decay 1e-4 \
  --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all"\
  --n-teleport 16 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1

#MNIST Teleport Muon
python3 src/train.py --seed 3 --dataset "MNIST" --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 128 --epochs 20 --d-model 128 --intermediate-size 512 --num-heads 4 --position-embedding "rope" \
  --opt "muon" --lr 0.015 --momentum 0.9 --weight_decay 1e-4 \
  --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all"\
  --n-teleport 16 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1

# #CIFAR-10 Teleport
python3 src/train.py --seed 3 --dataset "CIFAR10" --img-size 32 --num_channels 3 --num-classes 10 \
  --batch-size 256 --epochs 50 --d-model 192 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "sgd" --lr 0.005 --momentum 0.9 --weight_decay 1e-5 \
  --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --n-teleport 16 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1

# #CIFAR-10 Teleport Muon
python3 src/train.py --seed 3 --dataset "CIFAR10" --img-size 32 --num_channels 3 --num-classes 10 \
  --batch-size 256 --epochs 50 --d-model 192 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "sgd" --lr 0.005 --momentum 0.9 --weight_decay 1e-5 \
  --tele-att 1 --tele-mlp 0 --tele-opt 1 --tele-sign 0 --tele-layer "all" \
  --n-teleport 16 --tele-batch 4 --tele-cons 4 --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1