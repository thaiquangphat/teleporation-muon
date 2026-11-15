"""Pre-Newton--Schulz Nesterov criterion using Frobenius norms."""

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


def nesterov_frobenius_metrics_from_optimizer(
    optimizer: torch.optim.Optimizer,
    parameter_names: Optional[set[str]] = None,
) -> Dict[str, object]:
    groups = muon_groups(optimizer)
    beta = float(groups[0].get("momentum", 0.9))
    norms: List[float] = []
    names: List[str] = []
    for name, parameter, _ in iter_scored_muon_matrices(optimizer, parameter_names):
        direction = nesterov_direction(parameter, optimizer, beta)
        norm = float(torch.linalg.vector_norm(muon_matrix_view(direction), ord=2).item())
        norms.append(norm)
        names.append(name)
    return aggregate_norms(
        norms, names, "frobenius", "nesterov_corrected_pre_newton_schulz", beta
    )


__all__ = ["nesterov_frobenius_metrics_from_optimizer"]
