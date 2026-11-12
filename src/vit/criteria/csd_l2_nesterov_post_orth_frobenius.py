"""Post-orthogonalization Nesterov criterion using the Frobenius norm.

For each Muon matrix parameter and teleportation candidate T, this criterion
forms Muon's Nesterov direction, applies the same finite Newton--Schulz
zeropower transform and aspect-ratio multiplier as ``muon._muon_update``,
and scores the resulting update:

    V_T = beta**2 * B_previous_T + (1 - beta**2) * G_T
    O_T = MuonZeropower(V_T)
    S(T) = sqrt(sum_l ||O_l(T)||_F**2)

The criterion does not update optimizer state. It is used only to rank
teleportation candidates; candidate generation and commitment remain in
``teleport.py``.
"""

from __future__ import annotations

import math
from typing import List

import torch

from muon import _zeropower_via_newtonschulz5
from .common import aggregate_norms, nesterov_direction


def _muon_orthogonalized_update(direction: torch.Tensor) -> torch.Tensor:
    """Mirror shape handling and scaling in ``muon._muon_update`` exactly."""
    update = direction
    if update.ndim == 4:
        update = update.flatten(1)
    update = _zeropower_via_newtonschulz5(update)
    return update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5


def nesterov_post_orth_frobenius_metrics_from_optimizer(
    optimizer: torch.optim.Optimizer,
):
    """Score post-Newton--Schulz Muon directions by Frobenius norm."""
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

            # For batched matrix tensors, compute one Frobenius norm per
            # trailing matrix, matching the post-orth nuclear criterion.
            component_norms = torch.linalg.vector_norm(
                orthogonalized, ord=2, dim=(-2, -1)
            ).reshape(-1)
            for component_norm in component_norms:
                norm = float(component_norm.item())
                if not math.isfinite(norm):
                    raise ValueError(
                        "Post-orthogonalization Frobenius scoring encountered "
                        "a non-finite norm."
                    )
                norms.append(norm)

    return aggregate_norms(
        norms,
        "post_orthogonalization_frobenius",
        "nesterov_corrected_post_newton_schulz_muon_update",
    )
