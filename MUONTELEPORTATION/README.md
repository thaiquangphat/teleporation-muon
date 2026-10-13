# MUONTELEPORTATION

This folder reimplements the `MNISTCIFAR/` Vision Transformer and adds Muon-aligned Q/K teleportation. Treat `MNISTCIFAR/` as the source of truth for the ViT architecture, data pipeline, logging, and shell-script style.

## Setup

```bash
cd MUONTELEPORTATION
python -m pip install -r requirements.txt
```

## Tests

```bash
python -m pytest tests/test_muon_aware_teleportation.py -q
```

## Baselines

Plain Muon / SGD and the original naive teleportation from `MNISTCIFAR/src/teleport.py` remain separate methods. They are not replaced by the alignment search.

```bash
# Plain Muon
python src/train.py --seed 3 --dataset MNIST --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 128 --epochs 20 --d-model 128 --intermediate-size 512 --num-heads 4 \
  --position-embedding rope --opt muon --lr 0.015 --momentum 0.9 --weight_decay 1e-4 \
  --tele-method none --save_dir ./checkpoints/plain

# Naive MNISTCIFAR teleport
python src/train.py --seed 3 --dataset MNIST --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 128 --epochs 20 --d-model 128 --intermediate-size 512 --num-heads 4 \
  --position-embedding rope --opt muon --lr 0.015 --momentum 0.9 --weight_decay 1e-4 \
  --tele-method naive --n-teleport 16 --tele-batch 4 --tele-cons 4 \
  --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1 --save_dir ./checkpoints/naive
```

## Muon-alignment teleportation

Query/key rows are scaled by a positive diagonal `D` and `D^{-1}`. For RoPE, adjacent rotary-pair scales are tied. Candidates are ranked by the Muon alignment score on Q/K weight matrices with fresh/zero momentum, then accepted only if a cloned virtual optimizer step beats regular Muon on the same minibatch by `--tele-accept-tol`.

```bash
python src/train.py --seed 3 --dataset MNIST --img-size 28 --num_channels 1 --num-classes 10 \
  --batch-size 128 --epochs 20 --d-model 128 --intermediate-size 512 --num-heads 4 \
  --position-embedding rope --opt muon --lr 0.015 --momentum 0.9 --weight_decay 1e-4 \
  --tele-method muon_alignment --tele-candidates 6 --tele-batch 4 --tele-cons 4 \
  --tele-high 1.65 --tele-low 0.35 --tele-epoch-array 1 --tele-accept-tol 1e-6 \
  --save_dir ./checkpoints/muon_alignment
```

## Smoke test

```bash
bash script_test.sh
```

Each run writes `log.json`, `last.pt`, and `best.pt` under `checkpoints/<run_name>/`. Teleport decisions are stored in `log.json` under `telemetry`.

Run `bash scripts.sh` to execute each configured experiment with seeds 1, 2, and 3. After all runs complete, it writes plots of the per-epoch metric means and mean best validation accuracy to `plots/`.
