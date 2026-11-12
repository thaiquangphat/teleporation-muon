"""Post-Newton--Schulz Nesterov criterion using the nuclear norm."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

try:
    from ..muon import _zeropower_via_newtonschulz5
except ImportError:  # ``python src/lm/train.py`` import path
    from muon import _zeropower_via_newtonschulz5

from .common import aggregate_norms, muon_groups, nesterov_direction, parameter_name


def _muon_orthogonalized_update(direction: torch.Tensor) -> torch.Tensor:
    """Mirror shape handling and scaling in ``lm.muon._muon_update``."""
    update = direction
    if update.ndim == 4:
        update = update.flatten(1)
    update = _zeropower_via_newtonschulz5(update)
    return update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5


def post_orthogonalization_metrics_from_optimizer(
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Any]:
    groups = muon_groups(optimizer)
    beta = float(groups[0].get("momentum", 0.9))
    norms: List[float] = []
    names: List[str] = []
    for group in groups:
        for parameter in group["params"]:
            if parameter.ndim < 2 or parameter.grad is None:
                continue
            direction = nesterov_direction(parameter, optimizer, beta)
            orthogonalized = _muon_orthogonalized_update(direction).float()
            component_norms = torch.linalg.matrix_norm(
                orthogonalized, ord="nuc", dim=(-2, -1)
            ).reshape(-1)
            for component_norm in component_norms:
                norms.append(float(component_norm.item()))
                names.append(parameter_name(optimizer, parameter, len(names)))
    return aggregate_norms(
        norms, names, "post_orthogonalization_nuclear",
        "nesterov_corrected_post_newton_schulz_muon_update", beta
    )


__all__ = ["post_orthogonalization_metrics_from_optimizer"]
