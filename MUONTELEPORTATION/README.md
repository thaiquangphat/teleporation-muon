# MUONTELEPORTATION

This folder compares a locally implemented, configurable Vision Transformer
on MNIST or CIFAR-10 using Muon, AdamW, and SGD, with and without
Muon-alignment teleportation. The model is built from PyTorch layers, so its
patch embedding, attention, encoder, normalization, and classifier are
directly controlled here. Dataset loading and the custom Muon optimizer
follow the repository's `MNISTCIFAR` implementation.

## Setup

```bash
cd MUONTELEPORTATION
python -m pip install -r requirements.txt
```

## Six-experiment run

Run `scripts.sh` from this folder. It executes exactly six runs for the
selected dataset: three optimizer baselines followed by the same three
optimizers with `muon_alignment` teleportation and optimizer-state reset.
CIFAR-10 is the default; select MNIST with `DATASET=MNIST`.

```bash
bash scripts.sh
DATASET=MNIST SEED=7 EPOCHS=30 bash scripts.sh
```

The model size, teleport candidate count, interval, optimizer-specific
learning rates (`MUON_LR`, `ADAMW_LR`, and `SGD_LR`), and output directory can
be controlled with the environment variables defined in the script. SGD uses
the ViT reference rates of `0.015` for MNIST and `0.005` for CIFAR-10 by
default; the AdamW default is `3e-4` and Muon's is `0.015`. Each run writes
JSON metrics and its best checkpoint under
`checkpoints/<dataset>_six_experiments_seed<seed>/`. After all six runs
complete, the script plots validation accuracy and loss for each optimizer
and method.

## Model and teleportation behavior

The model uses convolutional patch embeddings, a class token,
pre-normalized multi-head self-attention and GELU MLP blocks, final layer
normalization, and a classification head. RoPE is the default positional
embedding (`POSITION_EMBEDDING=rope`); learned position embeddings are also
available. No external ViT implementation is used.

For each ViT attention block, the selected positive diagonal scaling `D`
transforms the query and key projections as `Q' = DQ` and `K' = D^-1 K`.
Therefore `Q'K'^T = QK^T`, preserving attention scores and model outputs up
to floating-point rounding. The identity candidate is always included.
With RoPE, candidate scales are tied within each adjacent rotary channel pair
and sampled independently per attention head, so they commute with the
position rotations and preserve the same invariance.
Candidate scoring runs a virtual optimizer step from fresh state for the
transformed query/key projections and ranks candidates by gradient alignment
with that optimizer's actual update. If a non-identity candidate beats
identity by `--tele-accept-tol`, it is applied to the live model and optimizer
state for the transformed query/key projections is reset.
