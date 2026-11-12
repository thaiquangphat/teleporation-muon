"""Gradient-norm criterion used by the legacy naive teleportation method."""

import torch.nn as nn


def gradient_l2_score(model: nn.Module) -> float:
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().float().norm(2).item()) ** 2
    return squared_norm ** 0.5
