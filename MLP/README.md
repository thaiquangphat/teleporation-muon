# MLP Teleportation Experiments

This project is a simplified version of the MNISTCIFAR setup, built around a small MLP instead of a ViT. It trains the same model on MNIST and CIFAR-10, with optional weight-teleportation layers applied during training to test their effect on optimization.

## Layout

- `src/model.py`: simple feed-forward MLP
- `src/muon.py`: Muon optimizer with an AdamW auxiliary parameter group
- `src/datamodule.py`: dataset loaders for MNIST and CIFAR-10
- `src/teleport.py`: teleport matrix generation and application
- `src/train.py`: training loop with optional teleportation
- `src/logger.py`: JSON metrics logger
- `scripts.sh`: run baseline and teleport experiments for both datasets, three optimizers, and three seeds
- `analyze_results.py`: aggregate run logs into CSV summaries and plots

## Quick start

```bash
cd MLP
python -m pip install -r requirements.txt
python src/train.py --dataset MNIST --epochs 5 --batch-size 128 --lr 1e-3
python src/train.py --dataset CIFAR10 --epochs 5 --batch-size 128 --lr 1e-3
```

Available optimizers are `adamw`, `sgd`, and `muon`. Muon applies its orthogonalized momentum update to hidden-layer weight matrices and uses AdamW for biases and the output classifier. Configure its learning rate and momentum with `--muon-lr` and `--muon-momentum`; defaults are `0.02` and `0.95`.

## Teleportation example

```bash
python src/train.py \
  --dataset MNIST \
  --n-teleport 8 \
  --tele-batch 32 \
  --tele-opt 1 \
  --tele-high 1.1 \
  --tele-low 0.9 \
  --tele-cons 16 \
  --tele-mlp 1 \
  --tele-layer all
```

## Multi-seed experiment sweep and analysis

Run all combinations of MNIST/CIFAR-10, AdamW/SGD/Muon, and baseline/teleport mode for three seeds (36 runs total):

```bash
bash ./scripts.sh
```

The defaults use seeds 42, 123, and 2024; 5 epochs; and 0 DataLoader workers. Override settings with environment variables, for example `SEEDS="7 17 27" EPOCHS=10 bash ./scripts.sh`.

After runs finish, read the JSON logs, export per-run and mean/std seed summaries, and save training/validation curve plots plus a training-time comparison:

```bash
python ./analyze_results.py --results-dir ./checkpoints --output-dir ./analysis
```

The analyzer writes `runs.csv`, `summary.csv`, `validation_accuracy.png`, `training_loss.png`, `validation_loss.png`, and `training_time.png`. The timing chart compares total logged training wall time and displays standard deviation across completed seeds.

## Notes

- The teleportation logic mirrors the same idea used in the ViT version: generate an invertible matrix, apply it to one layer, and apply its inverse to the neighboring layer so the intermediate representation remains consistent.
- The MLP is intentionally lightweight so the experiments are fast and easy to debug.
