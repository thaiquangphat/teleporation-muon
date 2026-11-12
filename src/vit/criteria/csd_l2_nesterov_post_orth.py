"""Post-orthogonalization Nesterov criterion using Muon's actual NS routine."""

import math
from typing import List

import torch

from muon import _zeropower_via_newtonschulz5
from .common import aggregate_norms, nesterov_direction


def _muon_orthogonalized_update(direction: torch.Tensor) -> torch.Tensor:
    """Mirror shape handling and aspect-ratio scaling in muon._muon_update."""
    update = direction
    if update.ndim == 4:
        update = update.flatten(1)
    update = _zeropower_via_newtonschulz5(update)
    return update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5


def post_orthogonalization_metrics_from_optimizer(optimizer: torch.optim.Optimizer):
    norms: List[float] = []
    for group in optimizer.param_groups:
        if not bool(group.get("use_muon", False)):
            continue
        beta = float(group.get("momentum", 0.95))
        if not math.isfinite(beta) or not (0.0 <= beta < 1.0):
            raise ValueError(f"Muon momentum must lie in [0, 1); got {beta}.")
        for parameter in group["params"]:
            if parameter.ndim < 2 or parameter.grad is None:
                continue
            direction = nesterov_direction(parameter, optimizer, beta)
            orthogonalized = _muon_orthogonalized_update(direction).float()
            component_norms = torch.linalg.matrix_norm(
                orthogonalized, ord="nuc", dim=(-2, -1)
            ).reshape(-1)
            for component_norm in component_norms:
                norm = float(component_norm.item())
                if not math.isfinite(norm):
                    raise ValueError("Post-orthogonalization scoring encountered a non-finite norm.")
                norms.append(norm)
    return aggregate_norms(
        norms, "post_orthogonalization_nuclear",
        "nesterov_corrected_post_newton_schulz_muon_update"
    )
