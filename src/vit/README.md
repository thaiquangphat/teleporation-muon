# ViT architecture

This folder contains the Vision Transformer training code for Muon teleportation experiments.

## Run CIFAR experiments

```bash
python src/vit/train.py \
  --dataset CIFAR100 \
  --data-dir ./data \
  --batch-size 256 \
  --epochs 10 \
  --d-model 192 \
  --intermediate-size 768 \
  --num-heads 3 \
  --position-embedding rope \
  --opt muon \
  --lr 0.02 \
  --momentum 0.9 \
  --weight_decay 1e-5 \
  --tele-method csd_l2_nesterov \
  --tele-candidates 6 \
  --tele-batch 4 \
  --tele-cons 4 \
  --save_dir ./checkpoints-vit/csd_l2_nesterov
```

Other supported datasets:

- `CIFAR10`
- `CIFAR100`
- `MNIST`
- `IMAGENET`

## Run ImageNet-1K

Prepare the dataset first:

```bash
bash data_imet.sh ./data/imagenet
```

Then run:

```bash
python src/vit/train.py \
  --dataset IMAGENET \
  --data-dir ./data/imagenet \
  --batch-size 64 \
  --epochs 10 \
  --opt muon \
  --tele-method none
```

## Teleport methods supported

- `none`
- `naive`
- `csd_l2`
- `csd_l2_nesterov`
- `csd_l2_nesterov_frobenius`
- `csd_l2_nesterov_post_orth`
- `csd_l2_nesterov_post_orth_frobenius`

Tuned hyperparameters are displayed in `../../scripts_vit.sh`.
