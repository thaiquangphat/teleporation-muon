"""Shared matrix and Nesterov-direction helpers for scoring criteria."""

import math
from typing import Any, Dict, List

import torch


def muon_matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    """Match the matrix flattening convention used by ViT Muon."""
    value = tensor.detach().float()
    if value.ndim == 2:
        return value
    if value.ndim >= 3:
        return value.flatten(1)
    raise ValueError(
        "A parameter in a Muon group is not matrix-valued; "
        f"received a tensor with shape {tuple(tensor.shape)}."
    )


def nesterov_direction(parameter, optimizer, beta: float) -> torch.Tensor:
    """Return the raw direction used by Muon's Nesterov interpolation."""
    gradient = parameter.grad.detach().float()
    previous = optimizer.state.get(parameter, {}).get("momentum_buffer")
    beta_sq = beta * beta
    if torch.is_tensor(previous):
        return previous.detach().float().mul(beta_sq).add(gradient, alpha=1.0 - beta_sq)
    return gradient.mul(1.0 - beta_sq)


def aggregate_norms(
    norms: List[float], norm_kind: str, momentum_direction: str
) -> Dict[str, Any]:
    if not norms:
        raise RuntimeError("No Muon matrix parameters received gradients; cannot score candidates.")
    squared_sum = sum(value * value for value in norms)
    return {
        "score": math.sqrt(squared_sum),
        "squared_score": squared_sum,
        "matrix_count": len(norms),
        "matrix_norms": norms,
        "max_matrix_norm": max(norms),
        "sum_matrix_norms": sum(norms),
        "norm_kind": norm_kind,
        "momentum_direction": momentum_direction,
    }
