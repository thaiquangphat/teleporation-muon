"""Pre-Newton--Schulz Nesterov criterion using Frobenius norms."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from .common import aggregate_norms, muon_groups, muon_matrix_view, nesterov_direction, parameter_name


def nesterov_frobenius_metrics_from_optimizer(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    groups = muon_groups(optimizer)
    beta = float(groups[0].get("momentum", 0.9))
    norms: List[float] = []
    names: List[str] = []
    for group in groups:
        for parameter in group["params"]:
            if parameter.ndim < 2 or parameter.grad is None:
                continue
            direction = nesterov_direction(parameter, optimizer, beta)
            norm = float(torch.linalg.vector_norm(muon_matrix_view(direction), ord=2).item())
            norms.append(norm)
            names.append(parameter_name(optimizer, parameter, len(names)))
    return aggregate_norms(
        norms, names, "frobenius", "nesterov_corrected_pre_newton_schulz", beta
    )


__all__ = ["nesterov_frobenius_metrics_from_optimizer"]
