"""Pre-Newton--Schulz Nesterov CSD-L2 criterion using the nuclear norm."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from .common import aggregate_norms, muon_groups, muon_matrix_view, nesterov_direction, parameter_name


def nesterov_nuclear_metrics_from_optimizer(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """Compute sqrt(sum_W ||beta^2 B_(t-1) + (1-beta^2)G_t||_*^2)."""
    groups = muon_groups(optimizer)
    beta = float(groups[0].get("momentum", 0.9))
    norms: List[float] = []
    names: List[str] = []
    for group in groups:
        for parameter in group["params"]:
            if parameter.ndim < 2 or parameter.grad is None:
                continue
            direction = nesterov_direction(parameter, optimizer, beta)
            norm = float(torch.linalg.matrix_norm(muon_matrix_view(direction), ord="nuc").item())
            norms.append(norm)
            names.append(parameter_name(optimizer, parameter, len(names)))
    return aggregate_norms(
        norms, names, "nuclear", "nesterov_corrected_pre_newton_schulz", beta
    )


# Backward-compatible metrics API used by the earlier LM criterion module.
def nesterov_csd_l2_metrics(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    metrics = nesterov_nuclear_metrics_from_optimizer(optimizer)
    return {
        **metrics,
        "matrix_nuclear_norms": metrics["matrix_norms"],
        "max_matrix_nuclear_norm": metrics["max_matrix_norm"],
        "sum_matrix_nuclear_norms": metrics["sum_matrix_norms"],
    }

__all__ = [
    "nesterov_nuclear_metrics_from_optimizer",
    "nesterov_csd_l2_metrics",
]
