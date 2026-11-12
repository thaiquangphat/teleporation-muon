"""CSD-L2 criterion using the next EMA momentum matrix and nuclear norm."""

import math
from typing import List

import torch

from .common import aggregate_norms, muon_matrix_view


def csd_l2_metrics_from_optimizer(optimizer: torch.optim.Optimizer):
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
            gradient = parameter.grad.detach().float()
            previous = optimizer.state.get(parameter, {}).get("momentum_buffer")
            if torch.is_tensor(previous):
                direction = previous.detach().float().mul(beta).add(gradient, alpha=1.0 - beta)
            else:
                direction = gradient.mul(1.0 - beta)
            norm = float(torch.linalg.matrix_norm(muon_matrix_view(direction), ord="nuc").item())
            if not math.isfinite(norm):
                raise ValueError("CSD-L2 encountered a non-finite nuclear norm.")
            norms.append(norm)
    return aggregate_norms(norms, "nuclear", "ema_momentum")
