"""NDS-based, function-preserving Q/K teleportation for the ViT Muon setup.

The score is the Rayleigh quotient z^T H z / ||z||^2, where z is the joint
Q/K Muon update direction and H is the Hessian of the minibatch loss with
respect to all selected Q/K weight matrices. The HVP is joint, so cross-layer
and Q/K cross-Hessian blocks are retained.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

try:  # Support both `python train.py` and package-style imports.
    from .muon import _muon_update
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from muon import _muon_update


def generate_tele_scheduler(tele_batch, number_of_batch, tele_opt, tele_cons):
    if tele_batch < 0 or number_of_batch < 0 or tele_batch > number_of_batch:
        raise ValueError(
            "tele_batch must lie between zero and the number of training batches."
        )
    if tele_cons <= 0:
        raise ValueError("tele_cons must be positive.")
    if tele_opt == 0:
        scheduler = [True] * tele_batch + [False] * (number_of_batch - tele_batch)
        random.shuffle(scheduler)
    elif tele_opt == 1:
        scheduler = [False] * number_of_batch
        count = 0
        groups = (tele_batch + tele_cons - 1) // tele_cons
        for i in range(groups):
            for j in range(tele_cons):
                index = (i * number_of_batch) // groups + j
                if index < number_of_batch and count < tele_batch:
                    scheduler[index] = True
                    count += 1
    else:
        raise ValueError(f"Unsupported tele_opt={tele_opt}; expected 0 or 1.")
    return scheduler


def validate_bounds(low: float, high: float) -> Tuple[float, float]:
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ValueError(
            f"Teleport bounds must satisfy 0 < low < high; got low={low}, high={high}."
        )
    return low, high


def gen_inv_matrix(n_head, d_head, tele_high, tele_low, tele_sign):
    size = d_head
    dtype = torch.float32
    matrix = torch.eye(size, dtype=dtype)
    matrix *= torch.rand(size, dtype=dtype) * (tele_high - tele_low) + tele_low
    if tele_sign == 1:
        mask = torch.randint(0, 2, (size,), dtype=dtype) * 2 - 1
        matrix[range(size), range(size)] *= mask
    indices = torch.randperm(size)
    matrix = matrix[indices, :]
    indices = torch.randperm(size)
    matrix = matrix[:, indices]
    matrix_inv = torch.transpose(matrix, -1, -2)
    matrix_inv = torch.where(matrix_inv != 0, 1.0 / matrix_inv, matrix_inv)
    return torch.block_diag(*[matrix] * n_head), torch.block_diag(*[matrix_inv] * n_head)


def gen_rope_inv_matrix(n_head, d_head, tele_high, tele_low, tele_sign):
    if d_head % 2 != 0:
        raise ValueError(f"d_head must be even for RoPE structure, got d_head={d_head}")
    dtype = torch.float32
    two_pi = 2.0 * torch.pi

    def make_head_blocks():
        blocks = []
        inverse_blocks = []
        thetas = torch.rand(d_head // 2, dtype=dtype) * two_pi
        rhos = torch.rand(d_head // 2, dtype=dtype) * (tele_high - tele_low) + tele_low
        if tele_sign == 1:
            signs = (torch.randint(0, 2, (d_head // 2,)) * 2 - 1).to(dtype)
            rhos = rhos * signs
        cos_values = torch.cos(thetas)
        sin_values = torch.sin(thetas)
        for cosine, sine, rho in zip(cos_values, sin_values, rhos):
            block = torch.tensor(
                [[rho * cosine, -rho * sine], [rho * sine, rho * cosine]],
                dtype=dtype,
            )
            rho_inv = 1.0 / rho
            inverse_block = torch.tensor(
                [[rho_inv * cosine, rho_inv * sine], [-rho_inv * sine, rho_inv * cosine]],
                dtype=dtype,
            )
            blocks.append(block)
            inverse_blocks.append(inverse_block)
        return torch.block_diag(*blocks), torch.block_diag(*inverse_blocks)

    matrices, inverses = zip(*(make_head_blocks() for _ in range(n_head)))
    return torch.block_diag(*matrices), torch.block_diag(*inverses)


def attention_blocks(model: nn.Module, tele_layer: str) -> List[nn.Module]:
    layers = list(model.vit.encoder.layer)
    if not layers:
        raise ValueError("The ViT model has no encoder layers.")
    if tele_layer == "all":
        return layers
    if tele_layer == "first":
        return [layers[0]]
    if tele_layer == "last":
        return [layers[-1]]
    raise ValueError(f"Unsupported tele_layer: {tele_layer}")


def qk_linear_modules(
    model: nn.Module, tele_layer: str
) -> List[Tuple[nn.Linear, nn.Linear]]:
    return [
        (block.attention.attention.query, block.attention.attention.key)
        for block in attention_blocks(model, tele_layer)
    ]


def qk_weight_parameters(model: nn.Module, tele_layer: str) -> List[nn.Parameter]:
    parameters: List[nn.Parameter] = []
    for query, key in qk_linear_modules(model, tele_layer):
        parameters.extend([query.weight, key.weight])
    return parameters


def _matrix_scale_summary(matrix: Optional[torch.Tensor]) -> List[float]:
    if matrix is None:
        return []
    return torch.linalg.vector_norm(matrix.float(), ord=2, dim=1).detach().cpu().tolist()


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def generate_symmetry_candidates(
    model: nn.Module,
    args: Any,
    low: float,
    high: float,
    count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate identity plus the source implementation's Q/K symmetries."""
    validate_bounds(low, high)
    if count < 2:
        raise ValueError("NDS needs at least two candidates, including identity.")

    q_linear = qk_linear_modules(model, args.tele_layer)[0][0]
    d_model = q_linear.out_features
    n_head = int(args.num_heads)
    if n_head <= 0 or d_model % n_head != 0:
        raise ValueError(
            f"Q/K output size {d_model} must be divisible by num_heads={n_head}."
        )
    d_head = d_model // n_head
    rope = getattr(args, "position_embedding", "learnable") == "rope"

    original_rng = _capture_rng_state()
    candidates: List[Dict[str, Any]] = []
    try:
        torch.manual_seed(int(seed))
        for index in range(count):
            if index == 0:
                matrix = torch.eye(d_model, dtype=torch.float32)
                matrix_inv = matrix.clone()
            elif rope:
                matrix, matrix_inv = gen_rope_inv_matrix(
                    n_head=n_head,
                    d_head=d_head,
                    tele_high=high,
                    tele_low=low,
                    tele_sign=int(getattr(args, "tele_sign", 0)),
                )
            else:
                matrix, matrix_inv = gen_inv_matrix(
                    n_head=n_head,
                    d_head=d_head,
                    tele_high=high,
                    tele_low=low,
                    tele_sign=int(getattr(args, "tele_sign", 0)),
                )
            candidates.append({
                "identity": index == 0,
                "M_att": matrix,
                "M_att_inv": matrix_inv,
                "attention_scale_summary": _matrix_scale_summary(matrix),
                "scale_min": min(_matrix_scale_summary(matrix), default=1.0),
                "scale_max": max(_matrix_scale_summary(matrix), default=1.0),
            })
    finally:
        _restore_rng_state(original_rng)
    return candidates


def apply_symmetry_candidate(
    model: nn.Module, candidate: Dict[str, Any], tele_layer: str
) -> None:
    if candidate.get("identity", False):
        return
    matrix = candidate["M_att"]
    matrix_inv = candidate["M_att_inv"]
    for query, key in qk_linear_modules(model, tele_layer):
        q_transform = matrix.to(device=query.weight.device, dtype=query.weight.dtype)
        k_inverse = matrix_inv.to(device=key.weight.device, dtype=key.weight.dtype)
        with torch.no_grad():
            query.weight.copy_(q_transform @ query.weight)
            if query.bias is not None:
                query.bias.copy_(query.bias @ q_transform.mT)
            key.weight.copy_(k_inverse.mT @ key.weight)
            if key.bias is not None:
                key.bias.copy_(key.bias @ k_inverse)


def _optimizer_groups_by_parameter(
    optimizer: torch.optim.Optimizer,
) -> Dict[int, Dict[str, Any]]:
    return {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _candidate_momentum_map(
    model: nn.Module,
    candidate_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    candidate: Dict[str, Any],
    tele_layer: str,
) -> Dict[int, torch.Tensor]:
    """Transport source Q/K momentum buffers into candidate coordinates."""
    source_pairs = qk_linear_modules(model, tele_layer)
    candidate_pairs = qk_linear_modules(candidate_model, tele_layer)
    momentum_by_candidate_id: Dict[int, torch.Tensor] = {}
    matrix = candidate["M_att"]
    matrix_inv = candidate["M_att_inv"]

    for (source_q, source_k), (candidate_q, candidate_k) in zip(
        source_pairs, candidate_pairs
    ):
        for source_parameter, candidate_parameter, transform in (
            (source_q.weight, candidate_q.weight, matrix_inv.mT),
            (source_k.weight, candidate_k.weight, matrix),
        ):
            state = optimizer.state.get(source_parameter, {})
            previous = state.get("momentum_buffer")
            if previous is None:
                transported = torch.zeros_like(candidate_parameter)
            elif candidate.get("identity", False):
                transported = previous.detach().clone().to(
                    device=candidate_parameter.device, dtype=candidate_parameter.dtype
                )
            else:
                transform_device = transform.to(
                    device=previous.device, dtype=previous.dtype
                )
                transported = (transform_device @ previous.detach()).to(
                    device=candidate_parameter.device,
                    dtype=candidate_parameter.dtype,
                )
            momentum_by_candidate_id[id(candidate_parameter)] = transported.clone()
    return momentum_by_candidate_id


def _candidate_group_map(
    model: nn.Module,
    candidate_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tele_layer: str,
) -> Dict[int, Dict[str, Any]]:
    """Map each cloned Q/K parameter to its source optimizer group."""
    source_parameters = qk_weight_parameters(model, tele_layer)
    candidate_parameters = qk_weight_parameters(candidate_model, tele_layer)
    if len(source_parameters) != len(candidate_parameters):
        raise RuntimeError("Q/K parameter structure changed in the candidate copy.")
    source_groups = _optimizer_groups_by_parameter(optimizer)
    mapped: Dict[int, Dict[str, Any]] = {}
    for source_parameter, candidate_parameter in zip(source_parameters, candidate_parameters):
        group = source_groups.get(id(source_parameter))
        if group is None:
            raise ValueError("A selected Q/K weight is missing from the optimizer.")
        mapped[id(candidate_parameter)] = group
    return mapped


def _compute_nds_score(
    candidate_model: nn.Module,
    candidate_groups: Dict[int, Dict[str, Any]],
    candidate_momentum: Dict[int, torch.Tensor],
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    tele_layer: str,
) -> float:
    """Compute the joint Q/K directional Hessian Rayleigh quotient."""
    parameters = qk_weight_parameters(candidate_model, tele_layer)
    if not parameters:
        raise ValueError("No Q/K projection weights were selected for NDS.")

    # HVPs require double backward through attention. Eager attention keeps this
    # path compatible with second-order autograd across supported backends.
    for module in candidate_model.modules():
        module_config = getattr(module, "config", None)
        if module_config is not None and hasattr(module_config, "_attn_implementation"):
            module_config._attn_implementation = "eager"

    logits = candidate_model(pixel_values=batch[0]).logits
    loss = criterion(logits, batch[1])
    gradients = torch.autograd.grad(
        loss, parameters, create_graph=True, retain_graph=True, allow_unused=False
    )

    directions: List[torch.Tensor] = []
    for parameter, gradient in zip(parameters, gradients):
        group = candidate_groups.get(id(parameter))
        if group is None or not bool(group.get("use_muon", False)):
            raise ValueError(
                "NDS requires every selected Q/K weight to belong to a Muon parameter group."
            )
        previous = candidate_momentum.get(id(parameter))
        if previous is None:
            previous = torch.zeros_like(parameter)
        direction = _muon_update(
            gradient.detach().clone(),
            previous.detach().clone(),
            beta=float(group.get("momentum", 0.95)),
            nesterov=bool(group.get("nesterov", False)),
        )
        if direction.shape != parameter.shape:
            direction = direction.reshape(parameter.shape)
        directions.append(direction.detach())

    denominator = sum(
        direction.double().square().sum() for direction in directions
    )
    if not torch.isfinite(denominator) or float(denominator.item()) <= 0.0:
        raise ValueError("The joint Q/K Muon direction has zero or non-finite norm.")

    directional_gradient = sum(
        (gradient.float() * direction.float()).sum()
        for gradient, direction in zip(gradients, directions)
    )
    hvp = torch.autograd.grad(
        directional_gradient,
        parameters,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    numerator = sum(
        (direction.double() * hessian_direction.double()).sum()
        for direction, hessian_direction in zip(directions, hvp)
    )
    score = (numerator / denominator).item()
    if not math.isfinite(score):
        raise ValueError(f"NDS produced a non-finite curvature score: {score}.")
    return float(score)


def _transport_live_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    candidate: Dict[str, Any],
    tele_layer: str,
) -> None:
    """Transport Muon Q/K momentum; clear transformed Adam bias state."""
    if candidate.get("identity", False):
        return
    matrix = candidate["M_att"]
    matrix_inv = candidate["M_att_inv"]
    for query, key in qk_linear_modules(model, tele_layer):
        for parameter, transform in (
            (query.weight, matrix_inv.mT),
            (key.weight, matrix),
        ):
            state = optimizer.state.get(parameter)
            if not state:
                continue
            momentum = state.get("momentum_buffer")
            if torch.is_tensor(momentum):
                transform_device = transform.to(
                    device=momentum.device, dtype=momentum.dtype
                )
                with torch.no_grad():
                    momentum.copy_(transform_device @ momentum)
        if query.bias is not None:
            optimizer.state.pop(query.bias, None)
        if key.bias is not None:
            optimizer.state.pop(key.bias, None)


def try_nds_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Choose the candidate with minimum NDS directional curvature.

    Candidate evaluation is virtual: each transformed model is a deep copy,
    source optimizer state is read-only, and the training RNG is restored after
    the search. Only the winning transformation and its momentum transport are
    committed to the live model/optimizer.
    """
    if not any(bool(group.get("use_muon", False)) for group in optimizer.param_groups):
        raise ValueError("NDS teleportation requires the Muon optimizer.")
    if not bool(getattr(args, "tele_att", 1)):
        raise ValueError("NDS currently scores Q/K attention projections only; set --tele-att 1.")

    low, high = validate_bounds(float(args.tele_low), float(args.tele_high))
    candidate_count = int(args.tele_candidates)
    if candidate_count < 2:
        raise ValueError("--tele-candidates must be >= 2 (identity plus at least one transform).")

    was_training = model.training
    rng_state = _capture_rng_state()
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
        results: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            _restore_rng_state(rng_state)
            candidate_model = copy.deepcopy(model)
            candidate_model.train(was_training)
            apply_symmetry_candidate(candidate_model, candidate, args.tele_layer)
            momentum_map = _candidate_momentum_map(
                model, candidate_model, optimizer, candidate, args.tele_layer
            )
            candidate_groups = _candidate_group_map(
                model, candidate_model, optimizer, args.tele_layer
            )
            with torch.enable_grad():
                score = _compute_nds_score(
                    candidate_model,
                    candidate_groups,
                    momentum_map,
                    batch,
                    criterion,
                    args.tele_layer,
                )
            results.append({
                "candidate_index": index,
                "identity": bool(candidate["identity"]),
                "nds_score": score,
                "scale_min": float(candidate["scale_min"]),
                "scale_max": float(candidate["scale_max"]),
            })
            del candidate_model, momentum_map, candidate_groups

        # Stable tie-breaking makes identity win exact ties.
        selected = min(
            results,
            key=lambda item: (
                item["nds_score"],
                0 if item["identity"] else 1,
                item["candidate_index"],
            ),
        )
        selected_index = int(selected["candidate_index"])
        selected_candidate = candidates[selected_index]
        identity_score = next(item["nds_score"] for item in results if item["identity"])
        selected_score = float(selected["nds_score"])
        accepted = not bool(selected["identity"])

        if accepted:
            apply_symmetry_candidate(model, selected_candidate, args.tele_layer)
            _transport_live_optimizer_state(
                model, optimizer, selected_candidate, args.tele_layer
            )
            committed = True

        return accepted, {
            "method": "nds",
            "accepted": accepted,
            "reason": "minimum_nds_score" if accepted else "identity_minimum_nds_score",
            "step": int(step_index),
            "score_definition": "z^T H_QK z / ||z||_2^2",
            "identity_score": float(identity_score),
            "selected_score": selected_score,
            "score_reduction": float(identity_score - selected_score),
            "selected_candidate_index": selected_index,
            "num_candidates_evaluated": len(results),
            "momentum_state_transported": accepted,
            "results": results,
        }
    finally:
        _restore_rng_state(rng_state)
        model.train(was_training)
