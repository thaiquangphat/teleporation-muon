#MNIST No teleport
python3 src/train.py --seed 3 --dataset "MNIST" --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 128 --epochs 1 --d-model 128 --intermediate-size 512 --num-heads 4 --position-embedding "rope" \
  --opt "sgd" --lr 0.015 --momentum 0.9 --weight_decay 1e-4