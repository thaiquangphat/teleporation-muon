"""Pre-orthogonalization Nesterov criterion using the Frobenius norm."""

import math
from typing import List

import torch

from .common import aggregate_norms, muon_matrix_view, nesterov_direction


def nesterov_frobenius_metrics_from_optimizer(optimizer: torch.optim.Optimizer):
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
            norm = float(torch.linalg.vector_norm(muon_matrix_view(direction)).item())
            if not math.isfinite(norm):
                raise ValueError("Nesterov Frobenius scoring encountered a non-finite norm.")
            norms.append(norm)
    return aggregate_norms(norms, "frobenius", "nesterov_corrected_pre_newton_schulz")
