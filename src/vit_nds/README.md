# ViT with NDS-based Muon teleportation

This folder is a separate implementation. It does not modify `src/vit/` and it
intentionally has no `criteria/` subfolder: the only teleportation score here
is the NDS directional-curvature score.

## Files

- `model.py`: the current uploaded ViT/RoPE model implementation.
- `teleport.py`: Q/K symmetry generation, virtual candidate evaluation, joint
  Hessian-vector products, candidate selection, and optimizer-state transport.
- `train.py`: training entry point based on the uploaded training script, with
  `--tele-method nds` as the only non-baseline teleportation method.
- `muon.py`: the Muon optimizer implementation used by this training variant.
- `datamodule.py`: the matching dataset loader implementation.

## NDS score

For each candidate symmetry, this implementation:

1. Deep-copies the current model and applies the function-preserving Q/K
   transformation.
2. Computes the current minibatch gradients at the transformed parameters.
3. Transports the existing Q/K Muon momentum buffers into the candidate
   coordinates, then forms the same EMA or Nesterov direction used by Muon.
4. Applies the same Newton--Schulz-5 transformation and aspect-ratio multiplier
   used by the optimizer to obtain each Q/K update matrix.
5. Concatenates all selected Q/K weight directions into one vector `z` and
   computes

   `NDS(candidate) = z^T H_QK z / ||z||_2^2`.

   The HVP is evaluated jointly over all selected Q/K weights. Therefore,
   Q/K and cross-layer Hessian blocks are included; the implementation does
   not score each projection independently.
6. Selects the candidate with the smallest score. Identity is included and
   wins exact ties. Only the selected transformation is committed to the live
   model. Its Muon momentum buffers are transported; transformed Q/K bias
   Adam states are cleared, consistent with the existing teleportation code.

`z` is the Muon matrix direction before multiplication by the learning rate
and before decoupled weight decay. Biases are transformed to preserve the
function, but are not part of `z` because those biases are in the auxiliary
Adam group rather than the Muon group.

For HVP compatibility, candidate evaluation uses eager attention. Training
continues with the model's configured attention backend. Candidate evaluation
uses the same input minibatch and restores the training RNG state afterward.

## Running

Run from the repository root, for example:

```bash
python src/vit_nds/train.py \
  --dataset CIFAR100 \
  --data-dir ../data \
  --num_channels 3 \
  --opt muon \
  --lr 0.01 \
  --momentum 0.9 \
  --position-embedding rope \
  --tele-method nds \
  --tele-layer all \
  --tele-candidates 6 \
  --tele-low 0.9 \
  --tele-high 1.1 \
  --tele-batch 32 \
  --tele-epoch-array 1
```

The `--tele-candidates` count includes the identity candidate. Thus, `6` means
one identity plus five non-identity transforms. NDS requires `--opt muon`.
The `--nesterov` flag controls whether both ordinary training and NDS candidate
scoring use the Nesterov-corrected Muon direction; without it, both use the
EMA momentum direction.

NDS is substantially more expensive than a first-order score: every candidate
requires a forward pass, first-order gradients, and a second-order HVP. Keep
`--tele-batch` and `--tele-candidates` modest for initial experiments.

The script retains the existing W&B configuration convention (`wandb.yaml`,
`WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`, and `WANDB_API_KEY`) and performs
only one W&B log write per epoch.
