"""Shared scoring utilities for LM teleportation criteria."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import torch
from torch import nn


def model_gradient_l2(model: nn.Module) -> float:
    """Global Euclidean norm of all available parameter gradients."""
    squared_norm = 0.0
    active = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        squared_norm += float(torch.sum(gradient * gradient).item())
        active += 1
    if active == 0:
        raise RuntimeError("No model gradients were produced during naive teleportation")
    score = math.sqrt(squared_norm)
    if not math.isfinite(score):
        raise ValueError("Naive teleportation produced a non-finite gradient norm")
    return score


def muon_groups(optimizer: torch.optim.Optimizer) -> list[dict]:
    """Return Muon groups and validate their shared momentum coefficient."""
    groups = [group for group in optimizer.param_groups if bool(group.get("use_muon", False))]
    if not groups:
        raise ValueError("CSD-Muon requires an optimizer group with use_muon=True")
    betas = [float(group.get("momentum", 0.9)) for group in groups]
    if any(abs(value - betas[0]) > 1e-12 for value in betas[1:]):
        raise ValueError("CSD scoring requires a common Muon momentum coefficient")
    if not math.isfinite(betas[0]) or not (0.0 <= betas[0] < 1.0):
        raise ValueError(f"Muon momentum must lie in [0, 1); got {betas[0]}.")
    return groups


def muon_matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return the float32 matrix view used by Muon criterion scoring."""
    value = tensor.detach().float()
    if value.ndim == 2:
        return value
    if value.ndim >= 3:
        return value.flatten(1)
    raise ValueError(f"Muon parameter must be matrix-valued, got {tuple(tensor.shape)}")


def ema_momentum_direction(
    parameter: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    beta: float,
) -> torch.Tensor:
    """Compute the next ordinary EMA momentum buffer without mutating state."""
    gradient = parameter.grad.detach().float()
    previous = optimizer.state.get(parameter, {}).get("momentum_buffer")
    if torch.is_tensor(previous):
        return previous.detach().float().mul(beta).add(gradient, alpha=1.0 - beta)
    return gradient.mul(1.0 - beta)


def nesterov_direction(
    parameter: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    beta: float,
) -> torch.Tensor:
    """Compute the actual pre-Newton--Schulz Muon Nesterov direction.

    The stored state is the previous EMA buffer B_(t-1). The upcoming step
    first forms B_t = beta B_(t-1) + (1-beta)G_t and then uses
    V_t = (1-beta)G_t + beta B_t = beta^2 B_(t-1) + (1-beta^2)G_t.
    """
    gradient = parameter.grad.detach().float()
    previous = optimizer.state.get(parameter, {}).get("momentum_buffer")
    beta_squared = beta * beta
    if torch.is_tensor(previous):
        return previous.detach().float().mul(beta_squared).add(
            gradient, alpha=1.0 - beta_squared
        )
    return gradient.mul(1.0 - beta_squared)


def aggregate_norms(
    norms: Iterable[float],
    matrix_names: Iterable[str],
    norm_kind: str,
    momentum_direction: str,
    beta: float,
) -> Dict[str, Any]:
    """Aggregate per-matrix norms into the candidate selection objective."""
    norm_values = [float(value) for value in norms]
    names = [str(value) for value in matrix_names]
    if not norm_values:
        raise RuntimeError("No Muon matrix parameters received gradients; cannot score candidates.")
    if not all(math.isfinite(value) for value in norm_values):
        raise ValueError(f"{norm_kind} criterion produced a non-finite matrix norm.")
    squared_sum = sum(value * value for value in norm_values)
    return {
        "score": math.sqrt(squared_sum),
        "squared_score": squared_sum,
        "matrix_count": len(norm_values),
        "matrix_norms": norm_values,
        "matrix_names": names,
        "max_matrix_norm": max(norm_values),
        "sum_matrix_norms": sum(norm_values),
        "norm_kind": norm_kind,
        "momentum_direction": momentum_direction,
        "beta": float(beta),
    }


def parameter_name(optimizer: torch.optim.Optimizer, parameter: nn.Parameter, index: int) -> str:
    """Read optional human-readable names installed by the LM optimizer factory."""
    return str(getattr(optimizer, "parameter_names", {}).get(id(parameter), f"parameter_{index}"))
