"""Naive global-gradient-L2 criterion for GPT-2 teleportation."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

try:
    from ..teleport import (
        apply_symmetry_candidate,
        generate_symmetry_candidates,
        restore_rng_state,
        capture_rng_state,
        transport_teleportation_optimizer_state,
        validate_bounds,
    )
except ImportError:  # ``python src/lm/train.py`` import path
    from teleport import (
        apply_symmetry_candidate,
        generate_symmetry_candidates,
        restore_rng_state,
        capture_rng_state,
        transport_teleportation_optimizer_state,
        validate_bounds,
    )

from .common import model_gradient_l2, muon_groups


def try_naive_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Select the best candidate if a majority beat identity on gradient L2.

    Candidate zero is identity. Each candidate is evaluated on a deep copy in
    eval mode, so the search does not mutate the live model or optimizer. If
    accepted, the selected symmetry is committed and the existing Muon
    momentum / affected Adam states are transported consistently.
    """
    started = time.perf_counter()
    rng_state = capture_rng_state()
    low, high = validate_bounds(args.tele_low, args.tele_high)
    candidate_count = int(getattr(args, "n_teleport", 6))
    if candidate_count < 2:
        raise ValueError("naive teleportation requires --n-teleport >= 2")
    muon_groups(optimizer)

    was_training = model.training
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
            inputs, targets = batch
            amp_dtype = getattr(args, "amp_dtype", "none")
            if inputs.device.type == "cuda" and amp_dtype == "bfloat16":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = candidate_model(input_ids=inputs).logits
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                logits = candidate_model(input_ids=inputs).logits
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            grad_l2 = model_gradient_l2(candidate_model)
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
            del candidate_model, logits, loss

        identity_result = next(item for item in results if item["identity"])
        baseline_score = float(identity_result["gradient_l2"])
        larger_candidates = [
            item for item in results
            if not item["identity"] and float(item["gradient_l2"]) > baseline_score
        ]
        best_candidate = max(results, key=lambda item: float(item["gradient_l2"]))
        majority_passed = len(larger_candidates) > candidate_count / 2.0
        accepted = bool(majority_passed and not best_candidate["identity"])
        selected_index = int(best_candidate["candidate_index"]) if accepted else 0
        selected_candidate = candidates[selected_index]
        selected_score = float(best_candidate["gradient_l2"] if accepted else baseline_score)

        if accepted:
            apply_symmetry_candidate(model, selected_candidate, args)
            transport_teleportation_optimizer_state(model, optimizer, selected_candidate, args)
            reason = "majority_of_random_candidates_exceeded_identity_gradient_l2"
        else:
            reason = "majority_gradient_l2_condition_not_met"

        return accepted, {
            "method": "naive",
            "accepted": accepted,
            "reason": reason,
            "step": int(step_index),
            "search_time_seconds": round(time.perf_counter() - started, 6),
            "selection_metric": "global_gradient_l2",
            "baseline_score": baseline_score,
            "selected_score": selected_score,
            "score_gain": selected_score - baseline_score,
            "num_candidates_exceeding_identity": len(larger_candidates),
            "majority_threshold": candidate_count / 2.0,
            "majority_condition_met": majority_passed,
            "proposed_candidate_index": int(best_candidate["candidate_index"]),
            "selected_candidate_index": selected_index if accepted else None,
            "selected_scale": selected_candidate["scale_summary"],
            "scale_min": float(selected_candidate["scale_min"]),
            "scale_max": float(selected_candidate["scale_max"]),
            "num_candidates_evaluated": len(results),
            "results": results,
        }
    finally:
        restore_rng_state(rng_state)
        model.train(was_training)
