#MNIST No teleport
python3 src/train.py --seed 3 --dataset "MNIST" --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 128 --epochs 20 --d-model 128 --intermediate-size 512 --num-heads 4 --position-embedding "rope" \
  --opt "sgd" --lr 0.015 --momentum 0.9 --weight_decay 1e-4

#MNIST No teleport Muon
python3 src/train.py --seed 3 --dataset "MNIST" --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 128 --epochs 20 --d-model 128 --intermediate-size 512 --num-heads 4 --position-embedding "rope" \
  --opt "muon" --lr 0.015 --momentum 0.9 --weight_decay 1e-4

# #CIFAR-10 No teleport
python3 src/train.py --seed 3 --dataset "CIFAR10" --img-size 32 --num_channels 3 --num-classes 10 \
  --batch-size 256 --epochs 50 --d-model 192 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "sgd" --lr 0.005 --momentum 0.9 --weight_decay 1e-5

# #CIFAR-10 No teleport Muon
python3 src/train.py --seed 3 --dataset "CIFAR10" --img-size 32 --num_channels 3 --num-classes 10 \
  --batch-size 256 --epochs 50 --d-model 192 --intermediate-size 768 --num-heads 3 --position-embedding "rope" \
  --opt "muon" --lr 0.005 --momentum 0.9 --weight_decay 1e-5