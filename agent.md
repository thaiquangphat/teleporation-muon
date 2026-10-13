# Agent Task: Reimplement the MNISTCIFAR ViT and Correct Muon Teleportation

## Goal

Inside `MUONTELEPORTATION/`, reimplement the Vision Transformer (ViT) used in `MNISTCIFAR/` and apply the Muon-alignment teleportation framework below. **Treat `MNISTCIFAR/` as the source of truth for the model architecture and project conventions.** Keep this task focused; do not change datasets, training budgets, augmentations, or optimizer hyperparameters just to improve results.

## 1. Rebuild and validate the ViT first

1. Inspect the model and training code in `MNISTCIFAR/` before editing. Reproduce its ViT architecture and forward pass in `MUONTELEPORTATION/`, including patch embedding, positional embeddings/RoPE, attention, normalization, MLP, residual connections, pooling, and classifier, as applicable to the reference implementation.
2. Match model configuration, tensor shapes, initialization, and relevant forward-pass behavior. Reuse `MNISTCIFAR/` data, training, logging, and shell-script conventions where appropriate.
3. Add a parity test: with equivalent configuration and copied compatible weights, compare reference and reimplemented model outputs/logits on the same deterministic inputs. Investigate and fix discrepancies before testing teleportation. Document anything that cannot be matched exactly.
4. Preserve the existing `MUONTELEPORTATION/` baseline and runnable workflow. Keep plain Muon and the previous naive teleportation as separate baselines; do not silently change their behavior.

## 2. Implement function-preserving Q/K teleportation

For each attention block, transform query and key coordinates reciprocally. With PyTorch `nn.Linear` weights stored as `[out_features, in_features]`, apply the scale to **rows** of the query weight and reciprocal scales to rows of the key weight; transform their biases consistently:

\[
W_Q'=DW_Q,\quad b_Q'=Db_Q,\qquad
W_K'=D^{-1}W_K,\quad b_K'=D^{-1}b_K.
\]

Use positive, finite, bounded scales and always include the identity candidate. For RoPE, tied scales must be equal within each adjacent rotary pair so that the scaling commutes with the rotary transformation. Confirm the reference model's exact tensor conventions before implementing. Verify that this transform preserves logits within numerical tolerance on fixed inputs; reject invalid scales. Do not apply it to parameters outside the intended Q/K projections.

## 3. Implement the intended Muon-alignment framework

For each candidate transformation \(g\), calculate gradients on the same current minibatch and score **only the affected query/key weight matrices** using the repository's actual Muon update rule with fresh/zero momentum for those weights:

\[
\mathcal A(g)=\sum_{\ell}\eta_{Q,\ell}\langle G_{Q,\ell}(g),U_{Q,\ell}(g)\rangle_F
+\sum_{\ell}\eta_{K,\ell}\langle G_{K,\ell}(g),U_{K,\ell}(g)\rangle_F.
\]

Here \(U\) must match the actual implementation in `src/muon.py` (momentum convention, Nesterov if used, Newton–Schulz, scaling, and dtype). Do not approximate it with an ideal polar factor, infer it from full-model parameter displacement, or include Adam-managed parameters in the Muon score. Use alignment only to rank candidates; it is not the acceptance criterion.

For each promising candidate, run a **full virtual optimizer step** on an isolated clone of the model and optimizer, resetting state only for the transformed Q/K weights and biases. Compute its post-step loss on the same minibatch. Separately compute a regular-Muon baseline virtual step from a clone that preserves the live optimizer state unchanged.

Accept a non-identity transform only if

\[
J_{\mathrm{candidate}} < J_{\mathrm{regular\ Muon}}-\varepsilon,
\]

where both \(J\)'s are actual post-step losses and \(\varepsilon\) is a configurable loss tolerance. Never accept a candidate merely because its alignment score exceeds identity's. If the identity-with-reset candidate wins, log it as a reset-only effect, not a successful teleportation.

When accepted, apply the transformation to the live model and reset state only for the affected Q/K parameters. Let the ordinary training loop perform the real optimizer step **exactly once**. On rejection, leave live parameters and optimizer state unchanged.

## 4. Correctness and isolation requirements

- Virtual evaluations must not mutate live parameters, optimizer buffers/counters, gradients, scheduler, data-loader state, or caller RNG state. Use cloned model/optimizer state and preserve/restore CPU and CUDA RNG states.
- Candidate and regular baseline comparisons must use the same batch, training/evaluation mode, and controlled stochastic behavior. Keep actual weight decay in the virtual optimizer step and loss evaluation, but exclude it from the pure alignment score.
- Clone optimizer state faithfully, preserving parameter groups and mapping state to the cloned parameters correctly.
- Ensure candidate order does not affect results. Include clear telemetry for alignment score, identity-reset loss, candidate post-step loss, regular-Muon post-step loss, acceptance/rejection reason, selected scales, state reset, and search time.

## 5. Required tests and experiments

Before full training, add/run focused tests proving:

1. The reimplemented ViT matches the `MNISTCIFAR/` reference outputs with equivalent weights/configuration.
2. Q/K teleportation preserves logits, including a RoPE test with tied pair scales and a test rejecting invalid pair scales.
3. The alignment score uses only Q/K weight matrices and the actual reset-state Muon direction.
4. Identity virtual-step behavior and optimizer-clone behavior are correct; virtual evaluation leaves all live state and RNG unchanged.
5. A candidate with better alignment but worse post-step loss than regular Muon is rejected; a candidate is accepted only when it beats the stateful baseline by more than the tolerance.
6. Rejection leaves live parameters/state untouched, and an accepted transform resets only the intended optimizer state before exactly one real training step.
7. Plain Muon and the existing naive teleportation still run; a short CPU/small-model smoke test produces logs/checkpoints.

Use the `MNISTCIFAR/` directory structure and shell-script style for new files/runs. Keep output directories separate so previous results are not overwritten. First validate correctness with a tiny smoke test, then run matched-seed comparisons against regular Muon. Report actual commands, tests and outcomes; do not claim a performance improvement unless measurements support it.

## Workflow

Inspect the repository and `MNISTCIFAR/` first, implement model parity, then implement and test teleportation correctness, then run experiments. Do not perform a broad refactor or hyperparameter sweep before the correctness tests pass. When finished, report changed files, model-parity evidence, tests/commands run, limitations, and metric locations.
