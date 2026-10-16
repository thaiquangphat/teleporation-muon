# MUONTELEPORTATION

This folder reimplements the `MNISTCIFAR/` Vision Transformer and adds Muon-aligned Q/K teleportation. Treat `MNISTCIFAR/` as the source of truth for the ViT architecture, data pipeline, logging, and shell-script style.

## Setup

```bash
cd MUONTELEPORTATION
python -m pip install -r requirements.txt
cp wandb.yaml.example wandb.yaml
```

Set `api_key`, `project`, and (optionally) `entity` in the ignored `wandb.yaml`.
The shell scripts load this file and export the W&B environment variables; secrets
are not stored in the scripts or sent as run configuration. Set `mode` to `offline`
or `disabled` when running without an online W&B account.
For `modal_app.py`, provide a Modal secret named `wandb` containing
`WANDB_API_KEY` and `WANDB_PROJECT` (and optionally `WANDB_ENTITY`); the local
YAML file is excluded from the uploaded project files.

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

Each run writes W&B metrics and a teleport-event table, plus `last.pt` and
`best.pt` under `checkpoints/<run_name>/`. Metrics include batch and epoch loss,
validation accuracy/loss, learning rates, throughput, teleport attempts and
acceptance, data/model setup, checkpoint, training, and validation durations.

Run `bash scripts.sh` to execute each configured experiment. After all runs
complete, it fetches W&B history and writes plots of the per-epoch metric means
and mean best validation accuracy to `plots/`.

## ImageNet-1K

The ImageNet loader expects the dataset to be organized as
`<data-dir>/train/<class>/...` and `<data-dir>/val/<class>/...`, matching
`IMAGENET/`. Set `IMAGENET_DATA_DIR` to the dataset root before running
`scripts.sh`; it defaults to `./data/imagenet`. Images are cropped/resized to
224 pixels and normalized with ImageNet mean and standard deviation.
