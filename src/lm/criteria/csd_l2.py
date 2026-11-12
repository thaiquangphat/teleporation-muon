"""Ordinary EMA-momentum CSD-L2 criterion using the nuclear norm."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from .common import aggregate_norms, ema_momentum_direction, muon_groups, muon_matrix_view, parameter_name


def csd_l2_metrics_from_optimizer(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """Compute sqrt(sum_W ||beta B_(t-1) + (1-beta)G_t||_*^2)."""
    groups = muon_groups(optimizer)
    beta = float(groups[0].get("momentum", 0.9))
    norms: List[float] = []
    names: List[str] = []
    for group in groups:
        for parameter in group["params"]:
            if parameter.ndim < 2 or parameter.grad is None:
                continue
            direction = ema_momentum_direction(parameter, optimizer, beta)
            norm = float(torch.linalg.matrix_norm(muon_matrix_view(direction), ord="nuc").item())
            norms.append(norm)
            names.append(parameter_name(optimizer, parameter, len(names)))
    return aggregate_norms(norms, names, "nuclear", "ema_momentum", beta)


__all__ = ["csd_l2_metrics_from_optimizer"]
