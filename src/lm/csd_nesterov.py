"""Nesterov CSD-L2 teleportation for the GPT-2 LM implementation.

This mirrors the current ViT ``csd_l2_nesterov`` selection rule while reusing
LM-specific GPT-2 symmetry transforms and optimizer-state transport.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

try:
    from .teleport import (
        _muon_groups,
        _muon_matrix_view,
        apply_symmetry_candidate,
        assert_live_state_unchanged,
        capture_rng_state,
        clone_model_and_optimizer,
        generate_symmetry_candidates,
        restore_rng_state,
        snapshot_live_state,
        transport_teleportation_optimizer_state,
        validate_bounds,
        _loss_for_batch,
    )
except ImportError:  # python src/lm/train.py
    from teleport import (
        _muon_groups,
        _muon_matrix_view,
        apply_symmetry_candidate,
        assert_live_state_unchanged,
        capture_rng_state,
        clone_model_and_optimizer,
        generate_symmetry_candidates,
        restore_rng_state,
        snapshot_live_state,
        transport_teleportation_optimizer_state,
        validate_bounds,
        _loss_for_batch,
    )


def nesterov_csd_l2_metrics(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """Score the exact raw Nesterov direction used before Muon's NS transform.

    With the repository's Muon update,

        B_t = beta B_{t-1} + (1-beta) G_t,
        V_t = (1-beta) G_t + beta B_t
            = beta^2 B_{t-1} + (1-beta^2) G_t.

    The CSD-L2 objective is

        sqrt(sum_W ||V_t(W)||_*^2),

    where the sum ranges over every matrix parameter assigned to Muon. No
    learning-rate weighting, polar factor, or Newton--Schulz iteration enters
    the candidate score.
    """
    groups = _muon_groups(optimizer)
    nuclear_norm_tensors: List[torch.Tensor] = []
    matrix_names: List[str] = []
    beta_values: list[float] = []

    for group in groups:
        beta = float(group.get("momentum", 0.9))
        if not math.isfinite(beta) or not (0.0 <= beta < 1.0):
            raise ValueError(f"Muon momentum must lie in [0, 1); got {beta}.")
        beta_values.append(beta)
        beta_squared = beta * beta
        gradient_weight = 1.0 - beta_squared

        for parameter in group["params"]:
            if parameter.ndim < 2 or parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            previous = optimizer.state.get(parameter, {}).get("momentum_buffer")
            if torch.is_tensor(previous):
                direction = previous.detach().float().mul(beta_squared).add(
                    gradient, alpha=gradient_weight
                )
            else:
                # First optimizer step: B_{t-1}=0.
                direction = gradient.mul(gradient_weight)

            nuclear_norm = torch.linalg.matrix_norm(_muon_matrix_view(direction), ord="nuc")
            nuclear_norm_tensors.append(nuclear_norm)
            parameter_name = getattr(optimizer, "parameter_names", {}).get(
                id(parameter), f"parameter_{len(matrix_names)}"
            )
            matrix_names.append(str(parameter_name))

    if not nuclear_norm_tensors:
        raise RuntimeError(
            "No Muon matrix parameters received gradients; cannot compute Nesterov CSD-L2"
        )
    if any(abs(value - beta_values[0]) > 1e-12 for value in beta_values[1:]):
        raise ValueError("CSD scoring requires a common Muon momentum coefficient")

    nuclear_norms = torch.stack(nuclear_norm_tensors).detach().cpu().tolist()
    if not all(math.isfinite(float(value)) for value in nuclear_norms):
        raise ValueError("Nesterov CSD-L2 produced a non-finite nuclear norm")
    squared_norm_sum = sum(float(value) ** 2 for value in nuclear_norms)
    return {
        "score": float(math.sqrt(squared_norm_sum)),
        "squared_score": float(squared_norm_sum),
        "matrix_count": len(nuclear_norms),
        "matrix_nuclear_norms": [float(value) for value in nuclear_norms],
        "matrix_names": matrix_names,
        "max_matrix_nuclear_norm": max(nuclear_norms),
        "sum_matrix_nuclear_norms": sum(nuclear_norms),
        "beta": beta_values[0],
        "momentum_direction": "nesterov_corrected_pre_newton_schulz",
    }




def _rankdata(values: List[float]) -> List[float]:
    """Average ranks (1-based) for Spearman correlation, including ties."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = 0.5 * ((cursor + 1) + end)
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _spearman_correlation(left: List[float], right: List[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = torch.tensor(_rankdata(left), dtype=torch.float64)
    y = torch.tensor(_rankdata(right), dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denominator.item() <= 1e-12:
        return None
    return float((torch.dot(x, y) / denominator).item())


def try_csd_l2_nesterov_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Select the exact finite-set maximizer of Nesterov CSD-L2.

    Candidate zero is the identity. Therefore identity wins ties, and a
    teleportation is accepted precisely when a non-identity candidate has a
    strictly larger score. The live optimizer/model are not modified during
    candidate evaluation; the selected symmetry is committed afterwards.
    """
    started = time.perf_counter()
    rng_state = capture_rng_state()
    live_snapshot = snapshot_live_state(model, optimizer)
    low, high = validate_bounds(args.tele_low, args.tele_high)
    candidate_count = int(args.tele_candidates)
    if candidate_count < 2:
        raise ValueError(
            "csd_l2_nesterov requires identity plus at least one alternative; "
            "set --tele-candidates >= 2"
        )
    muon_groups = _muon_groups(optimizer)
    if any(not bool(group.get("nesterov", False)) for group in muon_groups):
        raise ValueError(
            "Nesterov CSD-L2 requires every Muon parameter group to have nesterov=True; "
            "construct the optimizer with nesterov=True / pass --nesterov."
        )

    was_training = model.training
    committed = False
    try:
        model.train()
        candidates = generate_symmetry_candidates(
            model=model,
            args=args,
            low=low,
            high=high,
            count=candidate_count,
            seed=int(args.seed) + int(step_index),
        )
        if not candidates or not bool(candidates[0].get("identity", False)):
            raise RuntimeError("The LM candidate generator must put identity at index zero")

        results: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            candidate_model, candidate_optimizer = clone_model_and_optimizer(
                model, optimizer
            )
            # Every candidate sees the same dropout/RNG realization.
            restore_rng_state(rng_state)
            if not bool(candidate["identity"]):
                apply_symmetry_candidate(candidate_model, candidate, args)
                transport_teleportation_optimizer_state(
                    candidate_model, candidate_optimizer, candidate, args
                )

            candidate_optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_batch(candidate_model, batch, criterion, args)
            loss.backward()
            metrics = nesterov_csd_l2_metrics(candidate_optimizer)
            score = float(metrics["score"])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite Nesterov CSD-L2 score for candidate {index}")

            # Optional expensive diagnostic: run one virtual optimizer update
            # on each candidate clone, then evaluate the same batch in eval
            # mode. The live model/optimizer and candidate selection remain
            # untouched by this diagnostic rollout.
            post_step_loss = None
            if bool(getattr(args, "diagnostics_candidate_rollout", False)):
                candidate_optimizer.step()
                candidate_was_training = candidate_model.training
                candidate_model.eval()
                try:
                    with torch.no_grad():
                        post_step_loss = float(
                            _loss_for_batch(candidate_model, batch, criterion, args).detach().float().item()
                        )
                finally:
                    candidate_model.train(candidate_was_training)

            results.append(
                {
                    "candidate_index": index,
                    "identity": bool(candidate["identity"]),
                    "transform_type": candidate["transform_type"],
                    "scale": candidate["scale_summary"],
                    "scale_min": float(candidate["scale_min"]),
                    "scale_max": float(candidate["scale_max"]),
                    "attention_scale": candidate["attention_scale_summary"],
                    "mlp_scale": candidate["mlp_scale_summary"],
                    "alignment": score,  # compatibility with earlier W&B charts
                    "selection_score": score,
                    "csd_l2_score": score,
                    "csd_l2_squared_score": float(metrics["squared_score"]),
                    "csd_l2_matrix_count": int(metrics["matrix_count"]),
                    "csd_l2_max_matrix_nuclear_norm": float(
                        metrics["max_matrix_nuclear_norm"]
                    ),
                    "csd_l2_sum_matrix_nuclear_norms": float(
                        metrics["sum_matrix_nuclear_norms"]
                    ),
                    "csd_l2_matrix_nuclear_norms": metrics["matrix_nuclear_norms"],
                    "csd_l2_matrix_names": metrics["matrix_names"],
                    "csd_l2_momentum_direction": metrics["momentum_direction"],
                    "post_step_loss": post_step_loss,
                    "csd_l2_rank": None,
                    "alignment_rank": None,
                }
            )
            del candidate_model, candidate_optimizer, loss, metrics

        ranked = sorted(
            results,
            key=lambda item: (
                -float(item["csd_l2_score"]),
                int(item["candidate_index"]),
            ),
        )
        for rank, item in enumerate(ranked, start=1):
            item["csd_l2_rank"] = rank
            item["alignment_rank"] = rank

        selected = ranked[0]
        selected_index = int(selected["candidate_index"])
        selected_candidate = candidates[selected_index]
        identity_result = next(item for item in results if item["identity"])
        best_alternative = max(
            (item for item in results if not item["identity"]),
            key=lambda item: float(item["csd_l2_score"]),
        )

        # Check purity before changing the live model.
        restore_rng_state(rng_state)
        model.train(was_training)
        assert_live_state_unchanged(model, optimizer, live_snapshot)

        accepted = not bool(selected["identity"])
        if accepted:
            apply_symmetry_candidate(model, selected_candidate, args)
            transport_teleportation_optimizer_state(
                model, optimizer, selected_candidate, args
            )
            committed = True
            reason = "selected_max_csd_l2_nesterov_score"
        else:
            reason = "identity_selected_max_csd_l2_nesterov_score_noop"

        identity_score = float(identity_result["csd_l2_score"])
        selected_score = float(selected["csd_l2_score"])
        gain = selected_score - identity_score
        identity_post_loss = identity_result.get("post_step_loss")
        selected_post_loss = selected.get("post_step_loss")
        post_loss_gain = (
            float(identity_post_loss) - float(selected_post_loss)
            if identity_post_loss is not None and selected_post_loss is not None else None
        )
        candidate_post_losses = [item.get("post_step_loss") for item in results]
        score_post_loss_spearman = None
        best_post_step_candidate_index = None
        if all(value is not None for value in candidate_post_losses):
            actual_improvements = [float(identity_post_loss) - float(value) for value in candidate_post_losses]
            score_post_loss_spearman = _spearman_correlation(
                [float(item["csd_l2_score"]) for item in results], actual_improvements
            )
            best_post_step_candidate_index = min(
                results, key=lambda item: float(item["post_step_loss"])
            )["candidate_index"]
        sorted_scores = sorted((float(item["csd_l2_score"]) for item in results), reverse=True)
        score_margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        telemetry = {
            "method": "csd_l2_nesterov",
            "selection_metric": "nesterov_csd_l2",
            "baseline_score": identity_score,
            "selected_score": selected_score,
            "score_gain": gain,
            "accepted": bool(accepted),
            "reason": reason,
            "step": int(step_index),
            "search_time_seconds": round(time.perf_counter() - started, 6),
            "identity_csd_l2_score": identity_score,
            "selected_csd_l2_score": selected_score,
            "csd_l2_gain": gain,
            "identity_alignment": identity_score,
            "selected_alignment": selected_score,
            "alignment_gain": gain,
            "selected_candidate_index": selected_index,
            "selected_identity": bool(selected["identity"]),
            "selected_scale": selected["scale"],
            "scale_min": float(selected["scale_min"]),
            "scale_max": float(selected["scale_max"]),
            "selected_attention_scale": selected_candidate[
                "attention_scale_summary"
            ],
            "selected_mlp_scale": selected_candidate["mlp_scale_summary"],
            "identity_rank": int(identity_result["csd_l2_rank"]),
            "state_reset": False,
            "muon_momentum_reset": False,
            "auxiliary_adam_state_reset": bool(accepted),
            "momentum_state_transported": bool(accepted),
            "selection_objective": (
                "csd_l2_nesterov_euclidean_aggregate_of_matrix_nuclear_norms"
            ),
            "csd_l2_definition": (
                "sqrt(sum_over_muon_matrices(nuclear_norm(" 
                "beta^2*previous_momentum + (1-beta^2)*current_gradient)^2))"
            ),
            "momentum_direction": "nesterov_corrected_pre_newton_schulz",
            "csd_l2_matrix_count": int(selected["csd_l2_matrix_count"]),
            "csd_l2_matrix_nuclear_norms": selected[
                "csd_l2_matrix_nuclear_norms"
            ],
            "csd_l2_matrix_names": selected["csd_l2_matrix_names"],
            "tele_layer": args.tele_layer,
            "transform_family": selected_candidate["transform_type"],
            "uses_attention": bool(getattr(args, "tele_att", 1)),
            "uses_mlp": bool(getattr(args, "tele_mlp", 0)),
            "post_loss_gain": post_loss_gain,
            "identity_post_step_loss": identity_post_loss,
            "selected_post_step_loss": selected_post_loss,
            "best_post_step_candidate_index": best_post_step_candidate_index,
            "candidate_score_post_loss_spearman": score_post_loss_spearman,
            "candidate_score_margin": float(score_margin),
            "proposed_candidate_index": int(best_alternative["candidate_index"]),
            "accepted_candidate_index": selected_index if accepted else None,
            "num_candidates_evaluated": len(results),
            "results": results,
        }
        return accepted, telemetry
    finally:
        restore_rng_state(rng_state)
        model.train(was_training)
        if not committed:
            assert_live_state_unchanged(model, optimizer, live_snapshot)



def _model_gradient_l2(model: nn.Module) -> float:
    """Compute the global Frobenius norm of all available parameter gradients."""
    squared_norm = 0.0
    active = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        squared_norm += float(torch.sum(grad * grad).item())
        active += 1
    if active == 0:
        raise RuntimeError("No model gradients were produced during naive teleportation")
    score = math.sqrt(squared_norm)
    if not math.isfinite(score):
        raise ValueError("Naive teleportation produced a non-finite gradient norm")
    return score


def try_naive_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Apply the ViT-style majority-gradient-norm naive teleport heuristic.

    Candidate zero is identity and the remaining ``n_teleport - 1`` candidates
    are sampled GPT-2 symmetries. Each candidate is evaluated on a deep-copied
    model in eval mode, so dropout is disabled and the live model/optimizer are
    never touched during the search. As in ViT, a non-identity transform is
    committed only when strictly more than half of the random candidates have
    a larger global gradient L2 norm than identity.

    Unlike the old ViT naive implementation, the selected LM symmetry also
    transports the existing Muon momentum buffer and invalidates affected Adam
    states, preserving the optimizer state in the teleported parameterization.
    """
    started = time.perf_counter()
    rng_state = capture_rng_state()
    low, high = validate_bounds(args.tele_low, args.tele_high)
    candidate_count = int(getattr(args, "n_teleport", 6))
    if candidate_count < 2:
        raise ValueError("naive teleportation requires --n-teleport >= 2")
    _muon_groups(optimizer)

    was_training = model.training
    committed = False
    try:
        candidates = generate_symmetry_candidates(
            model=model,
            args=args,
            low=low,
            high=high,
            count=candidate_count,
            seed=int(args.seed) + int(step_index),
        )
        if len(candidates) != candidate_count or not bool(candidates[0].get("identity", False)):
            raise RuntimeError("The LM candidate generator must return identity at index zero")

        results: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            restore_rng_state(rng_state)
            candidate_model = copy.deepcopy(model)
            candidate_model.eval()
            if not bool(candidate["identity"]):
                apply_symmetry_candidate(candidate_model, candidate, args)

            candidate_model.zero_grad(set_to_none=True)
            loss = _loss_for_batch(candidate_model, batch, criterion, args)
            loss.backward()
            grad_l2 = _model_gradient_l2(candidate_model)
            results.append(
                {
                    "candidate_index": index,
                    "identity": bool(candidate["identity"]),
                    "gradient_l2": grad_l2,
                    "selection_score": grad_l2,
                    "scale_min": float(candidate["scale_min"]),
                    "scale_max": float(candidate["scale_max"]),
                    "transform_type": candidate["transform_type"],
                    "attention_scale": candidate["attention_scale_summary"],
                    "mlp_scale": candidate["mlp_scale_summary"],
                }
            )
            del candidate_model, loss

        identity_result = next((item for item in results if item["identity"]), None)
        if identity_result is None:
            raise RuntimeError("Naive teleportation candidate set is missing identity")
        baseline_score = float(identity_result["gradient_l2"])
        larger_candidates = [
            item for item in results
            if not item["identity"] and float(item["gradient_l2"]) > baseline_score
        ]
        best_candidate = max(results, key=lambda item: float(item["gradient_l2"]))

        # Match the original ViT heuristic: compare the number of improving
        # non-identity samples against half of the total candidate count.
        majority_passed = len(larger_candidates) > candidate_count / 2.0
        accepted = bool(majority_passed and not best_candidate["identity"])
        selected_candidate_index = (
            int(best_candidate["candidate_index"]) if accepted else 0
        )
        selected_candidate = candidates[selected_candidate_index]
        selected_score = float(best_candidate["gradient_l2"] if accepted else baseline_score)

        # Candidate evaluation only touched deep copies. Commit the symmetry to
        # the live model and transport Muon/Adam state together.
        if accepted:
            apply_symmetry_candidate(model, selected_candidate, args)
            transport_teleportation_optimizer_state(
                model, optimizer, selected_candidate, args
            )
            committed = True
            reason = "majority_of_random_candidates_exceeded_identity_gradient_l2"
        else:
            reason = "majority_gradient_l2_condition_not_met"

        gain = selected_score - baseline_score
        telemetry = {
            "method": "naive",
            "accepted": accepted,
            "reason": reason,
            "step": int(step_index),
            "search_time_seconds": round(time.perf_counter() - started, 6),
            "selection_metric": "global_gradient_l2",
            "baseline_score": baseline_score,
            "selected_score": selected_score,
            "score_gain": gain,
            "identity_gradient_l2": baseline_score,
            "selected_gradient_l2": selected_score,
            "gradient_l2_gain": gain,
            "num_candidates_exceeding_identity": len(larger_candidates),
            "majority_threshold": candidate_count / 2.0,
            "majority_condition_met": majority_passed,
            "proposed_candidate_index": int(best_candidate["candidate_index"]),
            "selected_candidate_index": selected_candidate_index if accepted else None,
            "selected_identity": not accepted,
            "selected_scale": selected_candidate["scale_summary"],
            "scale_min": float(selected_candidate["scale_min"]),
            "scale_max": float(selected_candidate["scale_max"]),
            "selected_attention_scale": selected_candidate["attention_scale_summary"],
            "selected_mlp_scale": selected_candidate["mlp_scale_summary"],
            "state_reset": False,
            "muon_momentum_reset": False,
            "auxiliary_adam_state_reset": accepted,
            "momentum_state_transported": accepted,
            "selection_objective": "viT_style_majority_gradient_l2",
            "tele_layer": args.tele_layer,
            "transform_family": selected_candidate["transform_type"],
            "uses_attention": bool(getattr(args, "tele_att", 1)),
            "uses_mlp": bool(getattr(args, "tele_mlp", 0)),
            "num_candidates_evaluated": len(results),
            "results": results,
        }
        return accepted, telemetry
    finally:
        restore_rng_state(rng_state)
        model.train(was_training)


def try_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Dispatch LM teleportation to naive gradient-L2 or Nesterov CSD-L2."""
    method = getattr(args, "tele_method", "none")
    if method in {"none", None}:
        return False, {"method": "none", "accepted": False, "step": int(step_index)}
    if method == "naive":
        return try_naive_teleportation(
            model, optimizer, batch, criterion, args, step_index
        )
    if method in {"csd_l2_nesterov", "muon_alignment"}:
        return try_csd_l2_nesterov_teleportation(
            model, optimizer, batch, criterion, args, step_index
        )
    raise ValueError(
        f"Unsupported LM teleportation method {method!r}; "
        "use 'none', 'naive', or 'csd_l2_nesterov'."
    )


__all__ = [
    "nesterov_csd_l2_metrics",
    "try_csd_l2_nesterov_teleportation",
    "try_naive_teleportation",
    "try_teleportation",
]
