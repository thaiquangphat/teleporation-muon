# AI Agent Instructions: Muon-Aware Teleportation Experiments

## Mission

Implement a reproducible research prototype that tests whether **choosing a function-preserving teleportation for the next Muon update** can outperform ordinary Muon. The repository already contains a teleportation implementation/reference file and an `MNISTCIFAR` experiment area. Inspect those first. **Mirror the existing `MNISTCIFAR` structure and conventions rather than inventing a new project layout.** Keep the initial work focused on MNIST/CIFAR; do not start by changing the ImageNet experiment.

The current observation motivating this work is that naively applying teleportation and then running Muon has not improved over plain Muon. The proposed research change is to select a teleportation based on the update Muon will actually make, not merely to teleport randomly or maximize the gradient norm.

## Non-negotiable repository rules

1. **Inspect before coding.** Read the repository tree, the whole `MNISTCIFAR` folder, the relevant reference file, and the existing implementations of Muon, teleportation, models, training, logging, and shell scripts. Do not guess filenames or duplicate a feature before checking whether it already exists.
2. **Use `MNISTCIFAR` as the template.** Put the experiment where it fits the current structure and follow its naming, CLI, configuration, logging, checkpoint, and result-directory conventions. If a separate variant folder is the established convention, create one that mirrors the existing folder. Otherwise extend the existing experiment with explicit flags. Avoid reorganizing unrelated files.
3. **Treat the repository's reference file and implementation as the source of truth** for the existing teleportation method. Read it and record its path and the relevant method/sections in the new README or experiment notes. Reuse existing code where appropriate. Do not silently change the semantics of the current baseline.
4. **Keep a clean baseline.** `Muon` with teleportation disabled must reproduce the existing plain-Muon path. Existing naive teleportation must remain available as a separate comparison. Do not change its behavior while adding the proposed method.
5. **No hidden changes.** Do not silently change the model, data split, batch size, optimizer parameter groups, Muon implementation, learning-rate schedule, weight decay, or augmentation to make a variant look better. Every intentional difference must be exposed in the CLI/config and logged.
6. **Make the first version small and testable.** Implement a random-candidate search and one-step virtual-update scoring first. Do not begin with differentiating through the optimizer, second-order gradients, or a large architecture change.
7. **No unnecessary dependencies.** Use the repository's existing environment and libraries unless a new dependency is essential and documented.

## Phase 0 — Repository reconnaissance

Before editing, inspect at least:

- The repository root and its README/configuration files.
- The complete `MNISTCIFAR` directory tree, including source modules and every relevant `*.sh` script.
- The reference paper/file mentioned by the user and the code it describes.
- The current teleportation entry point and exactly how candidate changes of basis, layer selection, bounds, and timing are implemented.
- The current Muon implementation: momentum/Nesterov update, matrix orthogonalization (including Newton–Schulz iterations), learning-rate handling, weight decay, parameter grouping, and optimizer-state layout.
- The current model definitions, training loop, evaluation loop, random-seed handling, checkpointing, metrics, and output paths.

Run or identify the smallest existing `MNISTCIFAR` smoke-test command. Record the command and expected output in notes before changing code. Preserve existing CLI names and meanings wherever possible. If an option such as `--tele-low`, `--tele-high`, `--tele-layer`, `--tele-start`, `--tele-limit`, or an equivalent already exists, reuse it instead of creating a conflicting option. Verify each flag's actual meaning from the code; do not assume it from its name.

At the end of reconnaissance, write a short implementation note identifying:

- The actual files that will be edited or added.
- The existing reference implementation that will be reused.
- How to launch a baseline MNIST/CIFAR run.
- How the optimizer state is represented and which parameters are managed by Muon versus another optimizer.

Then proceed with implementation; do not stop to ask the user questions when the repository gives enough information to make a reasonable choice.

## Research hypothesis and mathematical specification

### Function-preserving symmetry for a two-layer ReLU MLP

Start with a two-layer MLP whose hidden activation is ReLU:

$$
 f(x)=W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2.
$$

Use the matrix-storage convention of PyTorch `nn.Linear`: `W1` has shape `[hidden, input]` and `W2` has shape `[output, hidden]`. For a positive diagonal matrix

$$
D=\operatorname{diag}(s_1,\ldots,s_h),\qquad s_j>0,
$$

teleport with

$$
W_1'=DW_1,\quad b_1'=Db_1,\quad W_2'=W_2D^{-1},\quad b_2'=b_2.
$$

This preserves the function because ReLU is positively homogeneous: `ReLU(s*a) = s*ReLU(a)` for `s > 0`. In this storage convention, the rows of `W1` and elements of `b1` are scaled, while the columns of `W2` are inversely scaled. Confirm the actual model's tensor layout before coding. Apply this only to compatible adjacent affine/ReLU layers. Do not claim that the same transformation is an exact symmetry for GELU or arbitrary architectures.

At minimum, validate the transformation by comparing logits before and after teleportation on the same input. In the first prototype, use a small ReLU MLP with dropout disabled during the symmetry check. If extending the same operation to a deeper ReLU network, use reciprocal positive scaling on each adjacent pair of layers and correctly transform the intervening bias.

The corresponding gradient-coordinate relations, when `D` is fixed and the model function is the same, are

$$
G_{W_1}'=D^{-1}G_{W_1},\qquad
G_{b_1}'=D^{-1}G_{b_1},\qquad
G_{W_2}'=G_{W_2}D.
$$

Use these as numerical sanity checks; do not hard-code transformed gradients in place of autograd.

### Teleportation parameterization

Parameterize scales with log-scales `z`, where `s_j = exp(z_j)`. Require all scales to be positive and bounded. If the configured bounds satisfy `0 < low < 1 < high`, center each candidate's log-scales so that `sum(z) = 0` (equivalently, the product of the scales is one), while enforcing the configured bounds. A practical implementation can sample a random vector, subtract its mean, then shrink the vector as needed so every scale is within the valid bounds. Keep the identity candidate `z = 0`, `D = I` in every candidate pool. Validate invalid bounds and report a useful error.

Use the repository's existing teleportation bounds and candidate sampling convention if they are already suitable. Do not reinterpret existing `low`/`high` options; introduce explicitly named Muon-aware options only when needed.

## Methods to implement and compare

Keep the methods selectable through CLI/config. Reuse an existing naive method rather than implementing a second copy of it.

### Method A — `none`: plain Muon baseline

- No teleportation is performed.
- Follow the original training path as closely as possible.
- This run is the reference for deciding whether the new method helps.

### Method B — existing naive teleportation

- Invoke the existing teleportation path exactly as it exists in the repository.
- Preserve its original configuration and semantics.
- Give it a distinct method label and output directory so its results cannot overwrite the baseline or new method.

### Method C — `grad_norm`: gradient-norm-selected candidate (control)

For each function-preserving candidate, compute gradients in the candidate representation and score it with the exact gradient-norm criterion the reference method uses, if applicable. If the reference method has a different established score, document that and use it as the control instead of inventing a misleading approximation. Select the best candidate from the sampled pool, with identity included. This is a control for testing whether an ordinary gradient-based criterion is sufficient.

### Method D — `muon_alignment`: Muon-update alignment score

For candidate `D`, compute the gradients and the actual momentum-derived Muon update direction in the teleported coordinates. Let `U_i(D)` be the **actual update direction generated by the repository's Muon code** for parameter matrix `i`, including the configured momentum/Nesterov and matrix-orthogonalization behavior. For an update of the form `W_i <- W_i - lr_i * U_i`, rank candidates by

$$
S_{\mathrm{align}}(D)=\sum_i \eta_i\langle G_i(D),U_i(D)\rangle_F.
$$

Larger positive values are a better first-order predicted loss decrease. Use the actual parameters, learning rates, and update directions from this run. Do not replace Muon's finite Newton–Schulz update with an ideal polar factor in the implementation. Do not compute the score from the raw gradient alone when the actual optimizer uses momentum.

This score is a first-order proxy, not a guarantee that the actual loss will decrease. Include identity in the pool and do not accept a candidate that scores worse than identity, apart from a clearly defined numerical tolerance.

### Method E — `one_step_loss`: virtual Muon-step loss (primary method)

This is the main proposed method. For every candidate `D`:

1. Create an isolated copy of the model parameters in the candidate representation.
2. Compute gradients on the current training minibatch, using the same loss and model mode intended for the real update.
3. Create a temporary optimizer state for that candidate according to the configured state-handling mode described below.
4. Compute a **virtual optimizer step** using exactly the same optimizer implementation, parameter groups, momentum/Nesterov, matrix-orthogonalization, and learning rates as the real run.
5. Evaluate the virtual post-step loss on the chosen scoring batch.
6. Rank candidates by the lowest virtual post-step loss. Identity is always a candidate and is the fallback.

Choose the candidate with the lowest virtual post-step loss only when it beats the identity candidate by at least `accept_tol`; otherwise skip teleportation for that step. A zero or small tolerance is acceptable for the initial experiment, but must be configurable and logged.

The virtual step must **not mutate the real model, real optimizer, gradient buffers, scheduler, scaler, RNG state beyond explicitly controlled sampling, or data-loader state**. Prefer a pure helper that maps copied parameters plus copied optimizer state to an update. If this is impractical for the current optimizer, use a safe temporary cloned model/optimizer and prove with tests that the real objects are unchanged. Never call `optimizer.step()` on the real optimizer merely to score a candidate and then try to undo it.

For the first version, random search over a small pool is enough; do not differentiate through the optimizer. Use the current minibatch for the initial MVP, and make it possible to add an optional separate scoring/probe batch later to test whether candidate selection overfits its search minibatch. Label the scoring protocol in the logs.

## Momentum and optimizer-state ablation

Teleportation changes the parameter coordinates. The handling of optimizer state must therefore be explicit. Implement a `state_mode` (or reuse an equivalent existing option) with three controlled choices for the **new candidate-scoring and actual teleport event**:

1. **`keep`**: retain existing optimizer buffers unchanged after teleportation. This represents the naive state-handling option.
2. **`transport`**: transform state buffers that represent gradient/momentum coordinates consistently with the parameter transformation. For the two-layer example, the first-layer gradient-like state uses the inverse hidden scaling, and the second-layer gradient-like state uses the corresponding right scaling. Transform the hidden bias state when that bias is transformed. For Adam-like states, if they are in scope, transform first moments linearly and second/max moments with the squared coordinate scale. Only transform state whose semantics are understood; never blindly scale every tensor in an optimizer state dictionary.
3. **`reset`**: reset the relevant optimizer state for the transformed parameters according to the optimizer API. Do not reset unrelated parameters unless the experiment explicitly says so.

Treat `transport` as a hypothesis to test, not as a guaranteed invariant: Muon's matrix orthogonalization is generally not equivariant to arbitrary diagonal rescaling. Document exactly which buffers were transformed and why. For the first MVP, it is acceptable to implement and test state transport for Muon's matrix momentum and the relevant transformed bias only if the code's state semantics are clear; report any optimizer-managed parameters that are outside that implementation.

**Important implementation detail:** a momentum buffer produced by an EMA of gradients is gradient-like state, not a parameter tensor. Apply the gradient-coordinate transformation, not automatically the same transformation as the corresponding weight. Inspect the repository's Muon source to determine whether its state buffer stores raw momentum, a Nesterov-adjusted direction, an orthogonalized direction, or something else. Follow that implementation rather than assuming a particular buffer layout.

Run the `one_step_loss` and, if feasible, `muon_alignment` methods with `keep`, `transport`, and `reset`. If the existing reference code supports only one of these safely, implement the supported modes first and document the limitation rather than silently faking the other modes.

## Training-loop and interface requirements

Integrate the method into the existing training flow with minimal changes. Reuse the current data pipeline, evaluation utilities, checkpoint format, logging, and command-line parser. Names below are illustrative: reuse existing equivalents when present and avoid duplicate flags with conflicting meanings.

- `--tele-method`: `none`, `naive`, `grad_norm`, `muon_alignment`, `one_step_loss`.
- `--tele-state-mode`: `keep`, `transport`, `reset`.
- `--tele-candidates`: number of sampled candidates, not counting identity if the code's convention says otherwise; document the convention.
- `--tele-interval`: run candidate search every N optimizer steps.
- `--tele-start-step` and `--tele-limit`: optional start/stop window, reusing existing options if available.
- `--tele-low` and `--tele-high`: valid positive scale bounds, reusing existing options when compatible.
- `--tele-accept-tol`: minimum predicted or virtual loss improvement required to accept a candidate.
- `--tele-search-batch-size` or the repository's existing batch controls, only if candidate scoring genuinely needs a separate batch-size control.
- Existing random seed, dataset, model, batch size, learning rate, momentum, and output-directory options.

The flags must have clear help text and validation. Print the resolved configuration once at run start and save it beside the metrics. Do not add dozens of speculative knobs; start with the minimum needed to make each experiment reproducible.

### Recommended control flow

At a step where teleportation is scheduled:

1. Obtain the current minibatch and record the baseline model loss before teleportation.
2. Build the candidate set (always including identity) using the fixed run seed and deterministic local sampling where possible.
3. For each candidate, apply the exact function-preserving transform to a clone and score it with the selected criterion.
4. Verify identity is valid and select the best candidate.
5. If the best candidate fails the acceptance threshold, do not teleport.
6. Otherwise apply the selected transformation to the real model and transform/keep/reset the relevant real optimizer state exactly as configured.
7. Assert or log that the model output/data loss is unchanged by teleportation, to numerical tolerance, before the ordinary training update.
8. Perform the ordinary real training update exactly once using the established training loop.
9. Continue normal evaluation/checkpointing.

The one-step virtual loss calculation must simulate the same minibatch and update that the real loop will then perform. Avoid accidentally scoring with one batch but taking the actual step with a different batch unless the code explicitly implements and logs a separate probe-batch design. Ensure model train/eval mode and stochastic layers are handled consistently. For the symmetry unit test, disable dropout and other stochastic operations; do not use a noisy stochastic forward pass to claim exact invariance.

## Logging and experiment reproducibility

Log enough to determine whether the method helps and what it costs. Follow the existing logger format; add fields without breaking current analysis scripts.

At minimum record:

- method, state mode, seed, dataset/model, optimizer settings, teleport interval/window, candidate count, bounds, and acceptance tolerance;
- optimizer step, whether teleportation was attempted/accepted, selected candidate index, selected criterion score, identity score, and predicted improvement;
- scale statistics for the chosen candidate (minimum, maximum, mean log-scale, and optionally a compact list for a tiny MLP);
- loss before teleportation and loss immediately after teleportation, which should agree within numerical tolerance;
- virtual post-step loss for identity and the selected candidate when using `one_step_loss`;
- train/validation/test metrics supported by the current experiment;
- candidate-search wall-clock time and cumulative training time, so overhead is visible;
- checkpoint and metric paths.

Do not overwrite prior runs. Build on the existing output naming pattern and include method, state mode, and seed in the run identity. Save the resolved config and a brief method description (JSON or the repo's standard format). Ensure the existing `analyze_results.py` or equivalent can still run; extend it minimally if it needs to recognize the new method labels. Do not fabricate metrics or claim a performance gain unless an actual run supports it.

## Experiment matrix

Implement launch scripts that mirror the existing `MNISTCIFAR` shell-script style. The precise commands must use the repository's actual entry point and argument names discovered during reconnaissance.

Run at least these comparisons with the **same initial seed, model, data order, training budget, scheduler, and batch size** wherever possible:

| ID | Method | State mode | Purpose |
|---|---|---|---|
| E0 | `none` (plain Muon) | normal baseline | Establish the reference performance |
| E1 | existing `naive` teleportation | its established behavior | Reproduce the current approach |
| E2 | `grad_norm` | `keep` initially | Control for gradient-based candidate selection |
| E3 | `muon_alignment` | `keep` | Test update-alignment selection |
| E4 | `one_step_loss` | `keep` | Test direct virtual-step selection |
| E5 | `one_step_loss` | `transport` | Test gradient-coordinate state transport |
| E6 | `one_step_loss` | `reset` | Test momentum reset |

If compute allows, repeat E3 and E4 with all three state modes. Run at least three paired seeds for any result that will be used to claim improvement. First verify each method on a short, small MLP smoke test before running a full MNIST/CIFAR schedule. Do not make the first milestone depend on expensive GPU access.

Keep ordinary training budgets and evaluation protocols equal. Report both:

- performance against optimizer steps (to see whether the update is better), and
- performance against wall-clock time (to see whether candidate search overhead erases any gain).

A candidate-search method can be useful even if it declines many candidate teleports; acceptance/rejection frequency is itself a meaningful diagnostic.

## Required tests

Add tests in the existing test convention, or add a small test module if none exists.

1. **Function invariance:** for random inputs and positive bounded scales, logits before/after teleportation match within a documented tolerance.
2. **Loss invariance:** loss before/after the transform matches in deterministic evaluation mode.
3. **Bias handling:** if the model has hidden bias, verify it is scaled; output bias remains unchanged.
4. **Gradient-coordinate sanity check:** on a deterministic model and input, gradients computed with autograd in the transformed model approximately obey the formulas above.
5. **Identity candidate:** `D = I` produces the original parameters/logits and its virtual-update result matches the ordinary optimizer update to numerical tolerance.
6. **No virtual-step mutation:** scoring candidates leaves the real model parameters, optimizer state, scheduler state, and global step unchanged.
7. **State modes:** `keep`, `transport`, and `reset` modify only the intended state entries and can run at least one real training step afterward.
8. **Safe rejection:** if no candidate beats identity by `accept_tol`, the code skips teleportation and the run remains valid.
9. **Bounds:** sampled scales are positive, finite, within the configured bounds, and satisfy the product-one constraint within tolerance if that constraint is enabled.
10. **Reproducibility:** with the same seed and configuration, candidate generation and method decisions are reproducible to the extent allowed by the project's deterministic settings.
11. **Existing regression:** the original no-teleport path still runs; existing naive teleportation remains callable; existing shell scripts and result analysis are not broken.

Prefer fast CPU unit tests. Use a short training smoke test to check end-to-end CLI parsing, a few optimizer steps, checkpoint/metrics creation, and analysis-script compatibility.

## Suggested implementation order

1. Reconnaissance and baseline command.
2. A small reusable function for the exact ReLU hidden-unit scaling transform, with function-invariance tests.
3. A deterministic candidate generator with identity included and valid positive bounded scales.
4. A read-only virtual-step/scoring utility using the real optimizer semantics.
5. The `one_step_loss` selector and acceptance/rejection logic.
6. The `muon_alignment` selector using the actual Muon direction.
7. The `grad_norm` comparison, reusing the reference implementation's score when possible.
8. Explicit state modes and state-transport tests.
9. CLI integration, logging, checkpoints, and shell launch scripts matching `MNISTCIFAR` conventions.
10. CPU tests, short end-to-end smoke run, and result-analysis validation.
11. Run the controlled experiment matrix when the code passes tests; summarize results without overstating conclusions.

Do not implement every item in one large refactor. Keep changes modular and run tests after each milestone.

## Completion criteria

The implementation is complete when:

- The existing `MNISTCIFAR` structure and reference teleportation code have been inspected and used appropriately.
- Plain Muon and the original naive teleportation still run unchanged as baselines.
- The new optimizer-aware method can score candidates without mutating the real optimizer, then apply an accepted function-preserving transform and take exactly one normal real step.
- Candidate selection, state mode, bounds, cadence, acceptance tolerance, and random seeds are controllable and logged.
- The required invariance, virtual-state, and regression tests pass.
- Shell scripts following the existing `MNISTCIFAR` conventions launch the planned comparisons and keep output files separate.
- A short smoke test runs locally, with exact commands and results documented.
- A README/experiment note describes the hypothesis, equations, chosen reference file, commands, output locations, caveats, and how to compare methods.

## Final report to the user

At the end, return a concise implementation report containing:

1. Files added/modified and why.
2. The reference file and existing `MNISTCIFAR` conventions followed.
3. Exact commands for a CPU smoke test and for launching the experiment variants.
4. Tests executed and pass/fail outcomes.
5. Any incomplete state-handling mode or implementation limitation.
6. Where metrics/checkpoints are saved and how to analyze them.
7. A clear distinction between **implemented and tested** versus **not yet run at full training scale**. Do not claim that Muon-aware teleportation improves accuracy until the experiment results establish it.
