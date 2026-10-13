# Agent Task: Correct Muon-Alignment Teleportation and Baseline-Aware Acceptance

## Objective

Fix the implementation in `MUONTELEPORTATION/` so it implements the intended algorithm:

1. Generate **function-preserving query/key teleportations**.
2. Score candidates using the alignment between gradients of the affected Muon weight matrices and Muon's **actual reset-state update direction**.
3. Use this score only to identify promising candidates.
4. Evaluate promising candidates by a **virtual optimizer step** and measure the actual post-step minibatch loss.
5. Compare each candidate against a virtual **regular-Muon baseline that preserves the live optimizer state**.
6. Apply a non-identity teleportation only if its post-step loss beats that baseline by the configured tolerance.
7. Never mutate the live model or optimizer during candidate evaluation.

The current code does not yet implement steps 4–6 correctly. Treat this as a correctness task first, not a hyperparameter sweep. Do not change the dataset, architecture, baseline learning rates, training schedule, or augmentation to make the teleportation look better.

## Repository conventions

Work inside `MUONTELEPORTATION/` and use the existing `MNISTCIFAR/` folder as the structural reference. Follow its separation of training entry points, supporting modules under `src/`, shell scripts, and run-specific logs/checkpoints. Do not put everything in one new script.

Read these files before editing:

- `MUONTELEPORTATION/README.md`
- `MUONTELEPORTATION/scripts.sh`
- `MUONTELEPORTATION/src/train.py`
- `MUONTELEPORTATION/src/vit_alignment.py`
- `MUONTELEPORTATION/src/muon.py`
- `MUONTELEPORTATION/src/model.py`
- `MUONTELEPORTATION/src/logger.py`
- `MUONTELEPORTATION/src/datamodule.py`
- The corresponding training/logging conventions in `MNISTCIFAR/`.

Preserve the current runnable entry point (`bash scripts.sh`) and the existing output/logging conventions. Do not edit unrelated project folders.

---

## 1. Current implementation: concrete issues to fix

### 1.1 Candidate acceptance is not baseline-aware

In `src/vit_alignment.py`, `try_vit_alignment_teleportation` currently accepts the best candidate when its alignment score is larger than the identity candidate's score:

```python
accepted = best_score > identity_score + accept_tolerance
```

This is **not** the acceptance rule we want. It only establishes that the candidate's surrogate score is larger than the identity candidate's surrogate score. It does not establish that teleportation improves on ordinary Muon training.

Implement the following distinction:

- **Candidate-ranking objective:** Muon alignment.
- **Acceptance objective:** actual post-update loss compared against a virtual regular-Muon step that preserves the current optimizer state.

Never accept a teleportation merely because its alignment score is higher than identity's.

### 1.2 The current score sums over every model parameter

The current `_alignment_score` constructs `parameters_with_grad` from all parameters and sums

```python
(parameter.grad * (previous - parameter)).sum()
```

for every parameter after a virtual step. However, the teleportation modifies only the query/key projections. The stated Muon-alignment score is intended to use the gradients and Muon directions for the **affected query/key weight matrices**, not a sum dominated by unchanged patch embedding, value/output, MLP, normalization, and classifier parameters.

Refactor scoring so that, for each attention block `ell`, it uses only the query/key **weight matrices**:

\[
\mathcal A(g)
=\sum_{\ell}\eta_{Q,\ell}\langle G_{Q,\ell}(g),U_{Q,\ell}(g)\rangle_F
+\sum_{\ell}\eta_{K,\ell}\langle G_{K,\ell}(g),U_{K,\ell}(g)\rangle_F.
\]

Here `U` must be Muon's actual update direction under the existing implementation, computed with a **fresh/zero momentum buffer for the transformed query/key weights**. Include the learning rates consistently. Do not score the auxiliary Adam bias updates as if they were Muon matrix updates. The query/key biases are still transformed and their optimizer state is still reset for the candidate virtual step, but they are not part of the Muon-specific alignment sum.

The model's other parameter gradients are needed to take a complete virtual optimizer step and calculate post-step loss, but they must not be part of this Muon-specific ranking score.

### 1.3 Do not infer Muon alignment from the full post-step parameter displacement

The current method first performs a full optimizer step, then computes gradient dot displacement for all parameters. That displacement includes effects such as weight decay and the auxiliary optimizer. It is not the clean objective

\[
\langle G_{Q},U_{Q}\rangle_F+\langle G_{K},U_{K}\rangle_F.
\]

Implement a small, well-tested helper that computes Muon's exact update direction using the actual implementation in `src/muon.py` (including momentum convention, Newton–Schulz routine, matrix-shape scaling, and any dtype behavior). Prefer refactoring a non-mutating helper into `muon.py` over duplicating or approximating the update formula in `vit_alignment.py`.

For scoring, create a zero momentum buffer and compute the same update direction the live Muon optimizer would compute for that gradient with reset state. The helper must not mutate the real optimizer state. Confirm that the helper matches a one-step cloned optimizer update after removing decoupled weight decay and learning-rate scaling.

Do not replace the repository's actual Muon update with an ideal SVD/polar-factor implementation for the experiment. The polar-factor identity may be mentioned in documentation, but the experiment must score the implemented Muon update.

### 1.4 Identity candidate and state reset must be treated carefully

The candidate-ranking pool must contain the identity transformation. For a candidate virtual step:

- clone the model and optimizer;
- apply the candidate transformation to the clone;
- reset optimizer state only for the transformed query/key parameters;
- retain state for every other parameter;
- calculate gradients and take a full virtual optimizer step.

The regular-Muon baseline is different:

- clone the current model and optimizer;
- apply no transformation;
- do **not** reset any optimizer state;
- compute gradients and take the full virtual optimizer step.

The candidate identity-with-reset case is useful as a diagnostic for the effect of resetting state alone. It must not be reported as a successful nontrivial teleportation. Log it separately if it beats the stateful baseline.

Do not compare a reset-state identity candidate against a regular baseline that has also been reset. The baseline must retain the live momentum/state exactly as it stands at the current training step.

### 1.5 Current transformation applies one vector to all blocks

`apply_attention_teleportation` currently applies one shared scale vector to every attention block. This is function-preserving, provided the RoPE pairing constraint is obeyed, but it restricts the candidate family. First fix and test the current implementation without changing the architecture or invariance conditions. After the core tests pass, add per-layer candidate scales as a separately logged improvement if time permits; each attention block may have its own positive diagonal `D_ell` because its query/key transformation preserves that block's attention logits independently.

Do not mix this search-space expansion into the initial correctness patch unless it can be implemented and tested without destabilizing the base fix.

### 1.6 Keep the method name honest across optimizer types

`scripts.sh` currently runs the teleportation method with Muon, AdamW, and SGD. The Muon-specific formula above is meaningful for the Muon run. If preserving the six-run comparison:

- retain the three optimizer-only baselines;
- retain the three optimizer-plus-alignment-teleport runs for comparison;
- describe the generalized AdamW/SGD variants as **optimizer-update alignment**, not Muon alignment;
- label the Muon version as `muon_alignment/reset` in its configuration/metrics;
- record the actual optimizer name used to score each run.

It is acceptable to retain the existing CLI alias for compatibility, but the README, run name, and telemetry must not suggest that AdamW or SGD uses a Muon update. Do not silently make AdamW/SGD compute the Muon score.

---

## 2. Mathematical formulation to implement

### 2.1 Function-preserving transformation

The model uses `nn.Linear` query and key projections. For each attention block, the linear outputs are column-vector conceptually, and the transformation is

\[
Q_\ell' = D_\ell Q_\ell,\qquad K_\ell'=D_\ell^{-1}K_\ell.
\]

In PyTorch `nn.Linear`, whose weight has shape `[out_features, in_features]`, this means multiplying each output row of `query.weight` by the corresponding scale and each row of `key.weight` by the reciprocal scale. Apply the same scale to the matching biases:

\[
W_{Q,\ell}'=D_\ell W_{Q,\ell},\quad b_{Q,\ell}'=D_\ell b_{Q,\ell},
\]
\[
W_{K,\ell}'=D_\ell^{-1}W_{K,\ell},\quad b_{K,\ell}'=D_\ell^{-1}b_{K,\ell}.
\]

For standard attention, this preserves the logits because

\[
(D_\ell Q_\ell)(D_\ell^{-1}K_\ell)^\top
=Q_\ell K_\ell^\top.
\]

For this model's RoPE implementation, scale adjacent rotary pairs equally within each head. This makes `D_ell` commute with the rotation, preserving the attention logits up to floating-point error. Keep positive, finite scales only.

The current implementation scales all attention blocks with the same `D`. That is an acceptable first-stage search family; per-block `D_ell` is a later improvement.

### 2.2 Alignment score

For each candidate `g`, let `G_Q,ell(g)` and `G_K,ell(g)` be the gradients at the transformed parameters. Let `U_Q,ell(g)` and `U_K,ell(g)` be the actual update directions returned by the repository's Muon update rule when the corresponding momentum buffers start at zero. Score only the affected query/key matrix weights:

\[
\mathcal A(g)
=\sum_{\ell}\eta_{Q,\ell}
\langle G_{Q,\ell}(g),U_{Q,\ell}(g)\rangle_F
+\sum_{\ell}\eta_{K,\ell}
\langle G_{K,\ell}(g),U_{K,\ell}(g)\rangle_F.
\]

This is a first-order surrogate: it estimates loss decrease due to the Muon matrix updates. It is not the final acceptance test. Do not claim that larger alignment guarantees lower post-step loss.

### 2.3 Actual virtual post-step loss

Define `J_reset(g)` as the minibatch loss after one virtual optimizer step from the transformed parameters, where the optimizer state of the transformed query/key parameters has been reset but all other optimizer states have been cloned unchanged:

\[
J_{\mathrm{reset}}(g)
=\mathcal L_B\bigl(\operatorname{VirtualStep}
(T_g(\theta_t),\,s_t\text{ with Q/K state reset})\bigr).
\]

Define the regular baseline using the actual current state `s_t` without transformation or reset:

\[
J_{\mathrm{base}}
=\mathcal L_B\bigl(\operatorname{VirtualStep}(\theta_t,s_t)\bigr).
\]

Select a candidate from an alignment-ranked shortlist, e.g.

\[
\mathcal C_t=\operatorname{TopQ}_{g\in\mathcal G_t}\mathcal A(g),
\qquad
g_t^*=\arg\min_{g\in\mathcal C_t}J_{\mathrm{reset}}(g).
\]

Accept a **non-identity** teleportation only if

\[
J_{\mathrm{reset}}(g_t^*)<J_{\mathrm{base}}-\varepsilon.
\]

Use the same minibatch, same model mode, and controlled stochastic RNG for candidate and baseline evaluations. `epsilon` is a loss-unit tolerance, not an alignment-score tolerance. Do not compare alignment scores to `tele_accept_tol` after this change.

If the identity-with-reset candidate beats the regular baseline, log it as a reset-only effect. Do not count it as an accepted non-identity teleportation.

---

## 3. Required implementation changes

### 3.1 Refactor candidate evaluation into explicit helpers

Keep these responsibilities separated in `src/vit_alignment.py` (function names may differ, but behavior should be clear):

1. `generate_attention_scales(...)`: produce valid positive candidates, always including identity.
2. `apply_attention_teleportation(...)`: apply the exact weight-and-bias transformation.
3. `reset_attention_optimizer_state(...)`: reset state only for the selected Q/K weights and biases.
4. `clone_model_and_optimizer(...)`: deep-copy the model and faithfully clone parameter groups, hyperparameters, and every optimizer state entry to the cloned parameters.
5. `compute_muon_alignment(...)`: compute the score on Q/K weight matrices only, using actual reset-state Muon update directions.
6. `evaluate_virtual_step(...)`: on cloned objects, optionally transform, optionally reset only Q/K state, compute the batch gradient, run a full optimizer step, then measure the post-step loss on the same batch. Return score (when relevant), pre-step loss, post-step loss, and diagnostics.
7. `try_vit_alignment_teleportation(...)`: generate/evaluate candidates, evaluate a stateful baseline, rank candidates, apply the acceptance rule, and log telemetry. This function must leave the live optimizer/model untouched except for an accepted transformation and its intended state reset.

The virtual step must run on a complete model, not only the Q/K projections: the acceptance objective is the actual minibatch loss after the optimizer's full update. Other parameters update normally in the virtual clone using their copied optimizer states.

### 3.2 Exact post-step loss evaluation

For the baseline:

1. `deepcopy` the current live model.
2. Clone the optimizer state exactly without modifying the live optimizer.
3. Compute gradients on the supplied batch and take one optimizer step in the clone, with its current state preserved.
4. Re-evaluate the same minibatch after the virtual step and record `baseline_post_loss`.

For a candidate:

1. `deepcopy` the current live model and clone the optimizer.
2. Apply `D` to the cloned Q/K weights and biases.
3. Verify (or in tests prove) that its pre-step predictions/loss match the original model.
4. Reset state for the transformed Q/K weights and biases in the cloned optimizer only.
5. Compute gradients; compute alignment before stepping; then take the optimizer step in the clone.
6. Re-evaluate the same minibatch after the step and record `candidate_post_loss`.

Use the actual optimizer update and configured weight decay in the virtual step. Do not exclude weight decay from the **post-step loss evaluation**; it is part of the real training rule. Keep it separate from the pure Muon alignment score.

If runtime matters, compute the alignment score and virtual post-step loss during the same candidate pass. With the existing small candidate count, correctness is more important than premature optimization.

### 3.3 RNG and training/evaluation mode

Candidate and baseline comparisons must be reproducible and fair:

- Preserve the input batch exactly; do not resample augmentation per candidate.
- Preserve and restore CPU RNG state and all CUDA RNG states around virtual scoring, so each candidate and the baseline see equivalent stochastic behavior.
- Run gradient computation in the same mode as the actual training step (`model.train()` in the current training loop). Evaluate the post-step loss using a consistent mode/protocol for all candidates and baseline.
- Restore the caller's RNG state after virtual scoring so candidate exploration does not change the subsequent training randomness.
- Do not leave stale `.grad` tensors on the live model; all gradients should be zeroed on clones, and the normal training loop must clear live gradients before the actual step.

If the network uses dropout in the future, ensure the same random masks are used for comparable virtual gradient calculations or document a carefully controlled approximation. Current model dropout defaults are zero, but implement the RNG discipline anyway.

### 3.4 Candidate distribution

Keep the identity candidate explicitly included. Ensure RoPE candidates always use equal scales for each adjacent rotary channel pair within each attention head.

The current code samples scales uniformly in `[low, high]`, which is not symmetric in log-space. For a follow-up improvement (after basic correctness tests), sample `z=log(scale)` symmetrically around zero and set `scale=exp(z)`. Optionally center per-head log-scales to remove the redundant uniform rescaling component, while preserving the rotary-pair constraint. Log the exact sampling convention in configuration and README.

Do not increase scale bounds merely to force acceptance. Respect the existing `--tele-low` and `--tele-high` configuration unless changing the candidate sampler intentionally and documenting the change.

### 3.5 Optimizer naming and current six-run script

Preserve the ability to run `bash scripts.sh` and keep the baseline comparisons. However, clean up naming/telemetry:

- When `--opt muon`, report the method as `muon_alignment/reset`.
- When `--opt adamw` or `--opt sgd`, report the comparable generalized variant as `optimizer_alignment/reset`, since its score uses that optimizer's actual update, not Muon's.
- Do not label an AdamW or SGD run as using a Muon update.
- Keep baseline-only runs unchanged.
- Store `optimizer`, `tele_method`, `state_mode`, alignment scores, baseline post-step loss, candidate post-step loss, accepted/rejected flag, chosen scales, candidate index, and rejection reason in JSON telemetry.
- Log both **alignment improvement** and **post-step loss improvement**. Never use the former as a substitute for the latter.

Keep old command-line arguments working where reasonable, or provide a documented compatibility alias. Update `README.md` and shell script output to match actual semantics.

---

## 4. Acceptance and optimizer-state semantics

When the routine returns `accepted=True`, it must only apply the accepted transformation to the live model and reset the affected Q/K optimizer-state entries. The regular training loop should then perform the real optimizer step exactly once.

When the routine returns `accepted=False`:

- live model parameters must be byte-for-byte unchanged by virtual evaluations;
- live optimizer state must be unchanged;
- no momentum must be reset;
- the existing ordinary training step must proceed as before.

Do not update the live model using virtual candidate weights; virtual weights belong only to clones. Do not accidentally take a training step inside `try_vit_alignment_teleportation` and then take another in `train.py`.

For the accepted candidate's actual training step, transform and reset state immediately before the regular training gradient/update. Since the training loop currently calls the teleportation routine before computing the actual step, preserve that high-level ordering.

The baseline evaluator must clone the optimizer state **before** any candidate can mutate anything. Candidate order must not affect the baseline.

---

## 5. Tests: required before running full training

Use the repository's existing test layout if it has one; otherwise add focused tests under `MUONTELEPORTATION/tests/`. Keep test code isolated from training scripts.

### Test A: function-preservation test

For both `position_embedding='rope'` and `position_embedding='learnable'`:

- Build a small ViT with at least two heads and two encoder blocks.
- In `eval()` mode, compute logits on a fixed random batch.
- Apply a valid non-identity Q/K transform to a copied model.
- Compute logits again.
- Assert `torch.allclose(original_logits, transformed_logits, atol=..., rtol=...)` using sensible dtype-dependent tolerances.
- Verify that a deliberately invalid RoPE scale with unequal adjacent-pair entries is rejected.

This test verifies the actual transformation in the model, not only a standalone formula.

### Test B: gradient-coordinate sanity check

For a deterministic batch, confirm that transformed query/key gradients obey the expected coordinate relationships up to numerical tolerance. With the convention used in code, row-scaled query weights have reciprocal row scaling in their gradients, and row-scaled key weights have the opposite relation. Check biases consistently. If exact relation is awkward due parameter naming, at minimum compare gradients against a separately transformed clone and verify function invariance; document the convention carefully.

### Test C: optimizer clone isolation

Take snapshots of every live parameter and every tensor in `optimizer.state`. Run baseline virtual evaluation and several candidate virtual evaluations. Verify that:

- live model parameters are unchanged;
- live optimizer state is unchanged, including momentum, Adam moments, and step counters;
- parameter identities in the live optimizer groups are unchanged;
- no virtual gradient is left on the live parameters.

This test is essential; a failed candidate must have zero effect on training state.

### Test D: alignment objective uses only transformed Muon weights

Create a small deterministic model and a fixed batch. Compute the alignment score. Perturb only a parameter outside the Q/K projections or add a large artificial gradient to an unrelated parameter in a test-controlled scoring setup; the Muon alignment score must not change due to unrelated parameters. Verify that the score includes each transformed Q/K matrix exactly once and uses the actual zero-state Muon update helper.

Check the helper's computed Muon direction against the update recovered from a cloned optimizer step, with decoupled weight decay and learning-rate scaling accounted for.

### Test E: baseline-aware acceptance logic

Mock or construct controlled candidate evaluations with known alignment and post-loss values. Verify all of the following:

1. Higher alignment but worse post-step loss than regular Muon => reject.
2. Lower alignment but better post-step loss than regular Muon, provided it remains in the alignment shortlist => may be selected and accepted.
3. Candidate better than identity-reset but worse than stateful baseline => reject.
4. Identity-reset beats stateful baseline => log reset-only; do not count as non-identity teleportation.
5. Candidate post-loss is lower than stateful baseline by no more than tolerance => reject.
6. Candidate post-loss is lower by more than tolerance => accept, apply the transformation once, and reset only intended Q/K optimizer state.
7. Rejecting all candidates leaves the live model and optimizer state unchanged.

### Test F: same-batch fairness and reproducibility

With dropout enabled in a tiny test model (or a controlled stochastic module), verify that scoring order does not change scores and that running candidate evaluation does not advance/change the caller's RNG state. Candidate and baseline must use the same input tensors and controlled stochastic draws.

### Test G: tiny smoke run

Run a short training job for a small model with both:

- `--tele-method none`
- the corrected alignment method with `--opt muon`

Verify both complete and logs contain the expected fields. Also smoke-test the AdamW/SGD generalized alignment runs if the six-run script retains them.

---

## 6. Telemetry and debugging output

For every teleportation decision, record at least:

- global step and seed;
- optimizer name, method name, state mode;
- identity alignment score;
- each candidate's alignment score;
- each candidate's post-step loss where evaluated;
- baseline regular-optimizer post-step loss;
- identity-reset post-step loss;
- selected candidate index and scale range;
- accepted/rejected decision and a reason (`beats_baseline`, `no_alignment_gain`, `post_loss_not_better`, `identity_reset_only`, etc.);
- absolute improvement `baseline_post_loss - candidate_post_loss`;
- whether the transform was non-identity;
- elapsed search time and number of candidates evaluated.

Make the telemetry unambiguous: `alignment_gain` and `post_loss_gain` are distinct fields. An accepted candidate must satisfy the post-loss acceptance test, not merely the alignment test.

Avoid logging entire tensors or huge arrays for every candidate. Storing compact scale summaries plus the accepted vector is sufficient unless an optional debug flag is enabled.

---

## 7. Experiments and comparison protocol

Do not assume this fix makes teleportation better than regular Muon. The purpose is to make the code test the intended hypothesis faithfully.

After all tests pass:

1. Run the same seed, data order, batch size, model size, learning rate, momentum, weight decay, training schedule, and teleport interval for baseline and teleport variants.
2. Keep `scripts.sh`'s baseline runs intact.
3. Run the corrected Muon-alignment/reset variant and inspect acceptance frequency. If nearly every candidate is rejected, report that rather than relaxing acceptance until results look better.
4. Compare validation accuracy/loss and training time—not only the training minibatch loss used in candidate acceptance.
5. Repeat with multiple seeds only after the small smoke run succeeds.
6. Save outputs to unique run directories and never overwrite baseline checkpoints/logs.
7. Report the exact command, code changes, tests run, failures, and output paths in the final response.

Keep the first result table focused on:

- regular Muon;
- alignment-only selection as a diagnostic (if retained);
- corrected alignment + state reset + stateful-baseline post-loss acceptance;
- optional reset-only diagnostic.

Do not describe the previous alignment-only implementation as equivalent to the corrected algorithm.

---

## 8. Optional second-stage improvements (only after correctness is established)

### 8.1 Per-layer scales

Generalize candidates from a single vector `D` shared across all blocks to one `D_ell` per attention block. Each block's transformation is separately function-preserving. Use the same paired-channel constraint for RoPE and document the larger search cost. Keep an experiment flag so the shared-scale and per-layer-scale families can be compared directly.

### 8.2 Log-symmetric candidate sampling

Sample log-scales symmetrically around zero, `z ~ Uniform(-a, a)` and `scale = exp(z)`, instead of sampling scale values uniformly. For RoPE, tie adjacent entries. Consider centering the log-scales per head to remove redundant headwise scalar factors. Keep identity explicitly included.

### 8.3 Short rollout

Only after the one-step implementation and its tests work, optionally add a small `K`-step virtual rollout. Compare candidate and regular baseline trajectories using identical future minibatches and state semantics. A rollout candidate must beat the **stateful regular-Muon rollout**, not a baseline whose momentum was also reset. This is much more expensive; do not make it the first fix.

---

## 9. Definition of done

The task is complete only when:

- the candidate transformation is shown to preserve logits within tolerance;
- alignment is computed from affected Q/K **weight matrices** and the actual Muon update direction (for Muon runs);
- candidate selection uses alignment only as a proposal/ranking criterion;
- candidate post-step loss is evaluated after an actual virtual optimizer step;
- the baseline post-step loss is evaluated using the live optimizer state cloned without resetting it;
- acceptance requires a non-identity candidate to beat that stateful baseline by the configured loss tolerance;
- candidate evaluation cannot mutate live model/optimizer/RNG state;
- telemetry records both alignment and actual loss comparison;
- the `MNISTCIFAR`-style organization and existing runnable shell workflow are preserved;
- focused tests and a tiny smoke run pass;
- the final report explicitly states any remaining issues and does not claim performance improvements unless the experiment demonstrates them.
