"""Ordinary EMA-momentum CSD-L2 criterion using the nuclear norm."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .common import (
    aggregate_norms,
    ema_momentum_direction,
    iter_scored_muon_matrices,
    muon_groups,
    muon_matrix_view,
)


def csd_l2_metrics_from_optimizer(
    optimizer: torch.optim.Optimizer,
    parameter_names: Optional[set[str]] = None,
) -> Dict[str, object]:
    """Compute sqrt(sum_W ||beta B_(t-1) + (1-beta)G_t||_*^2)."""
    groups = muon_groups(optimizer)
    beta = float(groups[0].get("momentum", 0.9))
    norms: List[float] = []
    names: List[str] = []
    for name, parameter, _ in iter_scored_muon_matrices(optimizer, parameter_names):
        direction = ema_momentum_direction(parameter, optimizer, beta)
        norm = float(torch.linalg.matrix_norm(muon_matrix_view(direction), ord="nuc").item())
        norms.append(norm)
        names.append(name)
    return aggregate_norms(norms, names, "nuclear", "ema_momentum", beta)


__all__ = ["csd_l2_metrics_from_optimizer"]
