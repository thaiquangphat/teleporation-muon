import copy
import math
from typing import Any

import torch
from torch import nn


def validate_teleport_bounds(low: float, high: float, candidate_count: int) -> None:
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ValueError(f"Teleport bounds must satisfy 0 < low < high; got low={low}, high={high}.")
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1.")


def generate_attention_scales(
    hidden_size: int,
    low: float,
    high: float,
    candidate_count: int,
    seed: int,
    *,
    position_embedding: str = "learnable",
    num_heads: int = 1,
) -> list[torch.Tensor]:
    validate_teleport_bounds(low, high, candidate_count)
    if position_embedding not in {"rope", "learnable"}:
        raise ValueError("position_embedding must be 'rope' or 'learnable'.")
    if num_heads < 1 or hidden_size % num_heads:
        raise ValueError("hidden_size must be divisible by a positive num_heads.")
    head_size = hidden_size // num_heads
    if position_embedding == "rope" and head_size % 2:
        raise ValueError("RoPE requires an even attention head size.")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidates = [torch.ones(hidden_size, dtype=torch.float32)]
    for _ in range(candidate_count - 1):
        if position_embedding == "rope":
            paired_scales = torch.empty(
                num_heads, head_size // 2, dtype=torch.float32
            ).uniform_(low, high, generator=generator)
            scales = paired_scales.repeat_interleave(2, dim=-1).reshape(hidden_size)
        else:
            scales = torch.empty(hidden_size, dtype=torch.float32).uniform_(
                low, high, generator=generator
            )
        candidates.append(scales)
    return candidates


def _attention_linears(model: nn.Module) -> list[tuple[nn.Linear, nn.Linear]]:
    vit = getattr(model, "vit", None)
    encoder = getattr(vit, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise TypeError("Expected a ViT image-classification model with vit.encoder.layer.")

    linears = []
    for index, block in enumerate(layers):
        attention = getattr(getattr(block, "attention", None), "attention", None)
        query = getattr(attention, "query", None)
        key = getattr(attention, "key", None)
        if not isinstance(query, nn.Linear) or not isinstance(key, nn.Linear):
            raise TypeError(f"ViT encoder layer {index} does not expose linear query and key projections.")
        if query.out_features != key.out_features:
            raise ValueError(f"ViT encoder layer {index} has incompatible query/key projection sizes.")
        linears.append((query, key))
    if not linears:
        raise ValueError("The ViT model must contain at least one encoder attention layer.")
    return linears


def apply_attention_teleportation(model: nn.Module, scales: torch.Tensor) -> None:
    if not torch.isfinite(scales).all() or torch.any(scales <= 0):
        raise ValueError("Attention teleportation scales must all be finite and positive.")
    config = getattr(model, "config", None)
    if getattr(config, "position_embedding", None) == "rope":
        num_heads = config.num_attention_heads
        if scales.numel() % num_heads:
            raise ValueError("RoPE teleportation scales must divide evenly across attention heads.")
        paired_scales = scales.reshape(num_heads, -1, 2)
        if not torch.allclose(paired_scales[..., 0], paired_scales[..., 1]):
            raise ValueError("RoPE teleportation scales must be equal within each rotary channel pair.")
    for query, key in _attention_linears(model):
        if scales.numel() != query.out_features:
            raise ValueError(
                f"Expected {query.out_features} attention scales, received {scales.numel()}."
            )
        query_scale = scales.to(device=query.weight.device, dtype=query.weight.dtype)
        key_scale = query_scale.reciprocal().to(device=key.weight.device, dtype=key.weight.dtype)
        with torch.no_grad():
            query.weight.mul_(query_scale.unsqueeze(1))
            key.weight.mul_(key_scale.unsqueeze(1))
            if query.bias is not None:
                query.bias.mul_(query_scale)
            if key.bias is not None:
                key.bias.mul_(key_scale)


def reset_attention_optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    for query, key in _attention_linears(model):
        for parameter in (query.weight, query.bias, key.weight, key.bias):
            if parameter is not None:
                optimizer.state.pop(parameter, None)


def _clone_optimizer(
    model: nn.Module,
    model_clone: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.Optimizer:
    source_parameters = dict(model.named_parameters())
    clone_parameters = dict(model_clone.named_parameters())
    source_names = {id(parameter): name for name, parameter in source_parameters.items()}
    parameter_map = {}
    cloned_groups = []

    for group in optimizer.param_groups:
        cloned_group = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
        cloned_group["params"] = []
        for parameter in group["params"]:
            name = source_names.get(id(parameter))
            if name is None or name not in clone_parameters:
                raise ValueError("Optimizer contains a parameter that does not belong to the ViT model.")
            clone_parameter = clone_parameters[name]
            cloned_group["params"].append(clone_parameter)
            parameter_map[parameter] = clone_parameter
        cloned_groups.append(cloned_group)

    optimizer_clone = type(optimizer)(cloned_groups)
    for parameter, state in optimizer.state.items():
        clone_parameter = parameter_map.get(parameter)
        if clone_parameter is None:
            continue
        optimizer_clone.state[clone_parameter] = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for key, value in state.items()
        }
    return optimizer_clone


def _alignment_score(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    scales: torch.Tensor,
) -> float:
    model_clone = copy.deepcopy(model)
    optimizer_clone = _clone_optimizer(model, model_clone, optimizer)
    optimizer_clone.zero_grad(set_to_none=True)
    apply_attention_teleportation(model_clone, scales)
    reset_attention_optimizer_state(model_clone, optimizer_clone)

    loss = criterion(model_clone(pixel_values=images).logits, labels)
    loss.backward()
    parameters_with_grad = [
        parameter for parameter in model_clone.parameters() if parameter.grad is not None
    ]
    before_step = [parameter.detach().clone() for parameter in parameters_with_grad]
    optimizer_clone.step()

    score = 0.0
    with torch.no_grad():
        for parameter, previous in zip(parameters_with_grad, before_step):
            score += float((parameter.grad * (previous - parameter)).sum().item())
    return score


def try_vit_alignment_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    *,
    low: float,
    high: float,
    candidate_count: int,
    accept_tolerance: float,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    validate_teleport_bounds(low, high, candidate_count)
    if not math.isfinite(accept_tolerance) or accept_tolerance < 0:
        raise ValueError("accept_tolerance must be finite and non-negative.")
    hidden_size = _attention_linears(model)[0][0].out_features
    config = getattr(model, "config", None)
    candidates = generate_attention_scales(
        hidden_size,
        low,
        high,
        candidate_count,
        seed,
        position_embedding=getattr(config, "position_embedding", "learnable"),
        num_heads=getattr(config, "num_attention_heads", 1),
    )
    images, labels = batch

    identity_score = _alignment_score(
        model, optimizer, images, labels, criterion, candidates[0]
    )
    best_scale = candidates[0]
    best_score = identity_score
    candidate_scores = [{"candidate": 0, "alignment": identity_score, "identity": True}]

    for index, scales in enumerate(candidates[1:], start=1):
        score = _alignment_score(model, optimizer, images, labels, criterion, scales)
        candidate_scores.append({"candidate": index, "alignment": score, "identity": False})
        if score > best_score:
            best_scale = scales
            best_score = score

    accepted = best_score > identity_score + accept_tolerance
    info: dict[str, Any] = {
        "method": "muon_alignment",
        "state_mode": "reset",
        "accepted": accepted,
        "identity_alignment": identity_score,
        "selected_alignment": best_score,
        "step_seed": seed,
        "candidate_scores": candidate_scores,
    }
    if accepted:
        apply_attention_teleportation(model, best_scale)
        reset_attention_optimizer_state(model, optimizer)
        info["candidate"] = best_scale.tolist()
        info["scale_min"] = float(best_scale.min())
        info["scale_max"] = float(best_scale.max())
    else:
        info["candidate"] = None
    return accepted, info
