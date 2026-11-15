"""Naive global-gradient-L2 criterion for LM teleportation."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

try:
    from ..teleport import (
        apply_symmetry_candidate,
        capture_rng_state,
        generate_symmetry_candidates,
        restore_rng_state,
        transport_teleportation_optimizer_state,
        validate_bounds,
    )
except ImportError:  # ``python src/lm/train.py`` import path
    from teleport import (
        apply_symmetry_candidate,
        capture_rng_state,
        generate_symmetry_candidates,
        restore_rng_state,
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
    """Select a candidate if a strict majority beat identity on global grad L2."""
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
        if len(candidates) != candidate_count or not candidates[0]["identity"]:
            raise RuntimeError("The LM candidate generator must return identity at index zero")

        results: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            restore_rng_state(rng_state)
            candidate_model = copy.deepcopy(model)
            candidate_model.eval()
            if not candidate["identity"]:
                apply_symmetry_candidate(candidate_model, candidate, args)
            candidate_model.zero_grad(set_to_none=True)
            inputs, targets = batch
            if inputs.device.type == "cuda" and getattr(args, "amp_dtype", "none") == "bfloat16":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = candidate_model(input_ids=inputs).logits
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                logits = candidate_model(input_ids=inputs).logits
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            grad_l2 = model_gradient_l2(candidate_model)
            results.append({
                "candidate_index": index,
                "identity": bool(candidate["identity"]),
                "gradient_l2": grad_l2,
                "selection_score": grad_l2,
                "scale_min": float(candidate["scale_min"]),
                "scale_max": float(candidate["scale_max"]),
                "transform_type": candidate["transform_type"],
                "attention_scale": candidate["attention_scale_summary"],
                "mlp_scale": candidate["mlp_scale_summary"],
            })
            del candidate_model, logits, loss

        identity = next(item for item in results if item["identity"])
        baseline_score = float(identity["gradient_l2"])
        larger = [item for item in results if not item["identity"] and item["gradient_l2"] > baseline_score]
        best = max(results, key=lambda item: float(item["gradient_l2"]))
        majority_passed = len(larger) > candidate_count / 2.0
        accepted = bool(majority_passed and not best["identity"])
        selected_index = int(best["candidate_index"]) if accepted else 0
        selected_candidate = candidates[selected_index]
        selected_score = float(best["gradient_l2"] if accepted else baseline_score)
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
            "num_candidates_exceeding_identity": len(larger),
            "majority_threshold": candidate_count / 2.0,
            "majority_condition_met": majority_passed,
            "proposed_candidate_index": int(best["candidate_index"]),
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


__all__ = ["try_naive_teleportation"]
