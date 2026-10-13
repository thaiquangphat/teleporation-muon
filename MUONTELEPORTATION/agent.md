# Agent Instructions: Final Correctness Fix for Muon-Alignment Teleportation

## Goal

Review and correct the current implementation in `MUONTELEPORTATION/` so it implements **alignment-guided candidate selection plus stateful-baseline-aware acceptance**. Do not change the ViT architecture, dataset, or baseline hyperparameters to chase better metrics. Keep `MNISTCIFAR/` as the architecture and project-structure reference.

## First: inspect the active code path

Read these files before editing:

- `MUONTELEPORTATION/src/train.py`
- `MUONTELEPORTATION/src/teleport.py` — the current training entry point dispatches `try_teleportation` here
- `MUONTELEPORTATION/src/muon.py`
- `MUONTELEPORTATION/src/model.py`
- `MUONTELEPORTATION/tests/test_muon_aware_teleportation.py`
- `MUONTELEPORTATION/README.md`, `scripts.sh`, and `script_test.sh`
- The matching model/training files in `MNISTCIFAR/`

There is also a `src/vit_alignment.py` implementation that appears to contain an older alignment-only routine. Confirm whether anything imports it. Avoid maintaining two competing implementations: either remove/deprecate the stale duplicate or make the active entry point and documentation unambiguous. The code exercised by training and tests must be the same code.

## Main bug to fix: alignment is computed but not used for selection

In `src/teleport.py`, `try_muon_alignment_teleportation` computes an alignment score for every candidate, but then does:

```python
selected = min(results, key=_candidate_sort_key)
```

This selects the smallest virtual post-step loss across **all** random candidates. Alignment is only logged; it does not guide candidate selection. As written, the active method is effectively random-candidate one-step-loss search with a baseline gate, not the intended alignment-guided framework.

Implement the two-stage selector:

1. Generate candidates, always including identity.
2. Compute the Muon-alignment score for every candidate.
3. Rank candidates by descending alignment score.
4. Select a configurable shortlist of the top `q` **non-identity** candidates. Always evaluate identity separately as the reset-only diagnostic, even if its alignment score is not in the top `q`.
5. Evaluate virtual post-step loss only for identity and the shortlisted non-identity candidates.
6. Select the lowest-loss **non-identity** candidate from the shortlist, and accept it only if it beats the regular-Muon baseline by `tele_accept_tol`.
7. If no non-identity candidate beats the baseline, reject teleportation and preserve the live model and optimizer state. Log identity-reset loss separately; never report identity as a successful teleportation.

Add one explicit CLI argument such as `--tele-align-shortlist` (default a small value, e.g. `min(3, tele_candidates - 1)` after validation). Keep `--tele-candidates` semantics clear: currently the sampler includes identity in the total candidate count. Validate that shortlist size is positive and clamp or reject values greater than the number of non-identity candidates. Log the alignment rank and post-step loss for all evaluated candidates.

## Mathematical objective

For each attention layer `ell`, let `G_Q,ell(g)` and `G_K,ell(g)` be the gradients after a function-preserving Q/K transformation `g`. Let `U_Q,ell(g)` and `U_K,ell(g)` be the **actual repository Muon update directions computed with fresh/zero momentum** for those weight matrices. Rank by

\[
\mathcal A(g)=\sum_\ell \eta_{Q,\ell}\langle G_{Q,\ell}(g),U_{Q,\ell}(g)\rangle_F
+\sum_\ell \eta_{K,\ell}\langle G_{K,\ell}(g),U_{K,\ell}(g)\rangle_F.
\]

Continue using the existing `_muon_update` / Newton--Schulz implementation through a non-mutating zero-state helper; do not substitute an ideal polar factor. Score only Q/K **weight matrices** that belong to Muon's parameter group. Q/K biases are transformed and reset for virtual candidate updates, but they are not part of this Muon-specific alignment score.

The acceptance loss is different from the alignment score. Let `J_base` be the post-step loss of a full virtual optimizer step from a cloned model/optimizer with **all live optimizer state preserved**. Let `J_reset(g)` be the post-step loss of a cloned model transformed by `g`, with state reset only for the selected Q/K weights and biases and all other optimizer state preserved. Accept a non-identity candidate only if

\[
J_{\mathrm{reset}}(g^*) < J_{\mathrm{base}}-\varepsilon.
\]

Use the same input minibatch and the actual full optimizer update for both paths, including configured weight decay. Never use the alignment score as the acceptance threshold.

## Preserve these parts that are already on the right track

- The new `teleport.py` Q/K transform scales query rows by `D` and key rows by `D^{-1}`, including their biases.
- For RoPE, adjacent rotary-pair scales are tied, which is required for the diagonal transform to commute with the implemented rotary rotations.
- The candidate virtual-step path resets only the selected Q/K optimizer-state entries; other parameters retain their cloned state.
- The regular-Muon virtual baseline clones and preserves the live momentum state.
- The live optimizer is stepped only by the ordinary training loop, exactly once after any accepted transform.
- The strict state-dict/logit parity test against `MNISTCIFAR/src/model.py` should remain.

Do not regress these behaviors while fixing the selector.

## Additional correctness fixes

1. **Require Muon for `muon_alignment`.** Currently, the score skips parameters whose optimizer group lacks `use_muon`; with SGD/AdamW this can produce zero alignment scores while still running a candidate search. Either raise a clear error unless `--opt muon`, or implement a separately named generic optimizer-alignment method. Do not label a zero-score SGD/AdamW search as Muon alignment.
2. **Fair stochastic evaluation.** The current model config sets dropout to zero, but make candidate scoring robust: capture RNG state before the scoring phase, restore the same starting RNG state before each baseline/candidate evaluation, and restore the caller's RNG state and original model mode in a `try/finally` block. Candidate and baseline must see equivalent stochastic draws. Keep the current minibatch fixed; do not rerun data augmentation per candidate.
3. **Virtual-state isolation.** Preserve the existing snapshot assertions. Add checks that virtual scoring does not mutate the live optimizer's parameter-group hyperparameters, parameter identities, model buffers, `.grad` tensors, or RNG state. Exceptions during scoring must still restore the original RNG state and model mode.
4. **State cloning.** Keep optimizer cloning faithful to the source parameter groups and states. The current primary target is the repository's `SingleDeviceMuonWithAuxAdam`; test all Muon and auxiliary Adam state fields. If AdamW/SGD virtual scoring remains supported, preserve all parameter groups instead of assuming a single group, or explicitly scope the helper to the current supported optimizer.
5. **Honest logs.** Record per candidate: alignment score/rank, whether its virtual loss was evaluated, virtual post-step loss when available, and identity/non-identity. For each decision log `regular_muon_post_step_loss`, `identity_reset_loss`, best non-identity candidate loss, `post_loss_gain = regular_muon_post_step_loss - candidate_post_step_loss`, acceptance/rejection reason, and search time. Keep alignment gain and post-loss gain as separate fields.

## Required tests before running a full training job

Extend `tests/test_muon_aware_teleportation.py` with at least:

1. **Alignment shortlist is actually used:** mock scores and losses so the globally lowest-loss candidate is outside the top-`q` alignment shortlist; verify it is not evaluated/selected, and the best post-loss candidate inside the shortlist is selected.
2. **Alignment is not acceptance:** a high-alignment candidate with post-step loss worse than the stateful baseline must be rejected.
3. **Acceptance works:** a shortlisted non-identity candidate is accepted only if its post-step loss beats the stateful baseline by more than tolerance.
4. **Identity-reset diagnostic:** identity is evaluated separately; if reset-only wins, do not count it as teleportation and do not reset live state.
5. **No leakage:** candidate evaluation leaves live parameters, live optimizer states/groups, gradients, model buffers/mode, and caller RNG unchanged, on success and when an evaluation raises an exception.
6. **Actual Muon update:** compare the zero-state helper with a one-step cloned optimizer update for a Q/K matrix, accounting for learning rate and weight decay correctly.
7. **Model parity and invariance:** retain the strict-load/logit test against `MNISTCIFAR`; retain Q/K function-preservation and invalid RoPE-pair tests.
8. **Smoke test:** plain Muon and corrected `muon_alignment` both run through `script_test.sh` for a few CPU batches.

Run:

```bash
cd MUONTELEPORTATION
python -m pytest tests/test_muon_aware_teleportation.py -q
bash script_test.sh
```

Fix every failure rather than editing tests merely to pass. If a smoke command fails due to environment/dependency constraints, report the exact error and do not claim it passed.

## Experiment after correctness passes

Compare plain Muon against corrected alignment-shortlist/reset teleportation using the same seeds, model, minibatch order, optimizer settings, schedule, and compute budget. Keep the existing naive teleportation as a separate baseline. Report accuracy/loss against both optimizer steps and wall-clock time, plus attempted/accepted teleport counts. The candidate search can overfit the current minibatch even when its virtual loss is lower, so do not infer improved generalization from acceptance alone. Use paired multiple seeds before claiming an improvement.

## Final report

Return:

- Files changed and the active implementation path.
- The exact correction to candidate ranking and acceptance.
- Test commands and real pass/fail outputs.
- A short smoke-test result.
- Any unresolved limitation.
- A clear distinction between verified code correctness and unverified performance.

Do not claim that teleportation is better than regular Muon unless the experiments establish it.
