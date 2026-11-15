"""Pre-Newton--Schulz Nesterov CSD-L2 criterion using the nuclear norm."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .common import (
    aggregate_norms,
    iter_scored_muon_matrices,
    muon_groups,
    muon_matrix_view,
    nesterov_direction,
)


def nesterov_nuclear_metrics_from_optimizer(
    optimizer: torch.optim.Optimizer,
    parameter_names: Optional[set[str]] = None,
) -> Dict[str, object]:
    """Compute sqrt(sum_W ||beta^2 B_(t-1) + (1-beta^2)G_t||_*^2)."""
    groups = muon_groups(optimizer)
    beta = float(groups[0].get("momentum", 0.9))
    norms: List[float] = []
    names: List[str] = []
    for name, parameter, _ in iter_scored_muon_matrices(optimizer, parameter_names):
        direction = nesterov_direction(parameter, optimizer, beta)
        norm = float(torch.linalg.matrix_norm(muon_matrix_view(direction), ord="nuc").item())
        norms.append(norm)
        names.append(name)
    return aggregate_norms(
        norms, names, "nuclear", "nesterov_corrected_pre_newton_schulz", beta
    )


def nesterov_csd_l2_metrics(
    optimizer: torch.optim.Optimizer,
    parameter_names: Optional[set[str]] = None,
) -> Dict[str, object]:
    metrics = nesterov_nuclear_metrics_from_optimizer(optimizer, parameter_names)
    return {
        **metrics,
        "matrix_nuclear_norms": metrics["matrix_norms"],
        "max_matrix_nuclear_norm": metrics["max_matrix_norm"],
        "sum_matrix_nuclear_norms": metrics["sum_matrix_norms"],
    }


__all__ = ["nesterov_nuclear_metrics_from_optimizer", "nesterov_csd_l2_metrics"]
