# MUONTELEPORTATION

This prototype follows the research plan in `agent_instructions_muon_aware_teleportation.md` and mirrors the repository's `MNISTCIFAR` conventions: a small, deterministic training loop, JSON metrics logging, and simple shell scripts for repeated runs.

## Reference implementation reused

The source-of-truth baseline for the existing teleportation idea and optimizer structure is:

- `MNISTCIFAR/src/teleport.py` — the original generation of random invertible matrices and gradient-based teleportation checks
- `MNISTCIFAR/src/muon.py` — the Muon update logic (`SingleDeviceMuonWithAuxAdam` and `create_muon_optimizer`)

This new folder intentionally keeps the original baseline available and adds an explicit Muon-aware candidate selector rather than changing the existing behavior.

## Setup

```bash
cd MUONTELEPORTATION
python -m pip install -r requirements.txt
```

## Baseline run

```bash
python src/train.py --dataset MNIST --seed 1 --epochs 3 --batch-size 128 --opt muon --lr 0.015 --momentum 0.9 --weight-decay 1e-4
```

The baseline is the `--tele-method none` path. The `naive` MLP comparator
samples and applies a non-identity bounded ReLU scaling without scoring
candidate updates.

## New Muon-aware methods

The new prototype supports:

- `none`: plain Muon baseline
- `naive`: existing random teleportation logic
- `grad_norm`: control that compares candidate gradient norms
- `muon_alignment`: candidate score based on the repo's actual Muon update direction
- `one_step_loss`: primary method using a virtual Muon step and loss comparison

State handling is controlled with `--tele-state-mode`:

- `keep`
- `transport`
- `reset`

For `transport`, the hidden-layer Muon momentum is scaled by the inverse
hidden scale, hidden-bias Adam first/second moments by the inverse/squared
inverse scale, and output-layer Muon momentum by the hidden scale along its
input dimension. Unrelated state, including output-bias state, is unchanged.

Example:

```bash
python src/train.py --dataset MNIST --seed 1 --epochs 3 --batch-size 128 --opt muon \
  --tele-method one_step_loss --tele-state-mode transport --tele-candidates 8 \
  --tele-interval 1 --tele-low 0.7 --tele-high 1.5 --tele-accept-tol 1e-6
```

## Notes

- The function-preserving transform is the ReLU scaling symmetry for a two-layer MLP:
  - `W1' = D W1`, `b1' = D b1`, `W2' = W2 D^{-1}`
- Muon-aware candidate search includes the identity transform, and applies the selected transform only when it improves on identity by at least `--tele-accept-tol`.
- Candidate generation is deterministic for a fixed seed and logs all relevant configuration and selection stats in the run directory.

## Output

Each run writes JSON metrics and checkpoints under:

```text
MUONTELEPORTATION/checkpoints/<run_name>/
```

The folder contains the run config, log, last checkpoint, and best checkpoint.

## Run the experiment matrix

From this directory, run:

```bash
bash run_all_experiments.sh
```

The script runs MNIST and CIFAR-10 with one seed (default `1`) and 30 epochs,
first running the plain Muon baseline and then all four teleportation methods
with each state mode. Once every training command succeeds, it plots validation
accuracy/loss curves and best validation accuracy under
`checkpoints/all_settings_seed1/figures/`.

Defaults can be overridden with environment variables, for example:

```bash
SEED=7 EPOCHS=30 DATASETS="MNIST" bash run_all_experiments.sh
```

Install the plotting dependency with `python -m pip install -r requirements.txt`
before running if it is not already available.
