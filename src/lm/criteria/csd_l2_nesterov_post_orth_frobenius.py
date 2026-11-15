"""Post-Newton--Schulz Nesterov criterion using Frobenius norms."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

try:
    from ..muon import _zeropower_via_newtonschulz5
except ImportError:  # ``python src/lm/train.py`` import path
    from muon import _zeropower_via_newtonschulz5

from .common import (
    aggregate_norms,
    iter_scored_muon_matrices,
    muon_groups,
    nesterov_direction,
)


def _muon_orthogonalized_update(direction: torch.Tensor) -> torch.Tensor:
    update = direction
    if update.ndim == 4:
        update = update.flatten(1)
    update = _zeropower_via_newtonschulz5(update)
    return update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5


def nesterov_post_orth_frobenius_metrics_from_optimizer(
    optimizer: torch.optim.Optimizer,
    parameter_names: Optional[set[str]] = None,
) -> Dict[str, object]:
    groups = muon_groups(optimizer)
    beta = float(groups[0].get("momentum", 0.9))
    norms: List[float] = []
    names: List[str] = []
    for name, parameter, _ in iter_scored_muon_matrices(optimizer, parameter_names):
        direction = nesterov_direction(parameter, optimizer, beta)
        orthogonalized = _muon_orthogonalized_update(direction).float()
        component_norms = torch.linalg.vector_norm(
            orthogonalized, ord=2, dim=(-2, -1)
        ).reshape(-1)
        for component_norm in component_norms:
            norms.append(float(component_norm.item()))
            names.append(name)
    return aggregate_norms(
        norms, names, "post_orthogonalization_frobenius",
        "nesterov_corrected_post_newton_schulz_muon_update", beta
    )


__all__ = ["nesterov_post_orth_frobenius_metrics_from_optimizer"]
