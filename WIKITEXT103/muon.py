"""Single-device Muon optimizer adapted from Keller Jordan's Muon repository.

Copyright (c) 2024 Keller Jordan

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch


def _zeropower_via_newtonschulz5(gradient, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    update = gradient.to(torch.bfloat16)
    transposed = update.size(-2) > update.size(-1)
    if transposed:
        update = update.mT

    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = update @ update.mT
        update = a * update + (b * gram + c * (gram @ gram)) @ update

    return update.mT if transposed else update


def _muon_update(gradient, momentum_buffer, beta):
    momentum_buffer.lerp_(gradient, 1 - beta)
    update = gradient.lerp(momentum_buffer, beta)
    if update.ndim == 4:
        update = update.flatten(1)
    update = _zeropower_via_newtonschulz5(update)
    return update * max(1, update.size(-2) / update.size(-1)) ** 0.5


def _adam_update(gradient, exp_avg, exp_avg_sq, step, betas, eps):
    beta1, beta2 = betas
    exp_avg.lerp_(gradient, 1 - beta1)
    exp_avg_sq.lerp_(gradient.square(), 1 - beta2)
    corrected_avg = exp_avg / (1 - beta1**step)
    corrected_avg_sq = exp_avg_sq / (1 - beta2**step)
    return corrected_avg / (corrected_avg_sq.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Apply Muon to matrix weights and AdamW to other parameters."""

    def __init__(self, param_groups):
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Every Muon optimizer parameter group needs 'use_muon'.")
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                if group["use_muon"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    update = _muon_update(
                        parameter.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                    )
                else:
                    if not state:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        state["step"] = 0
                    state["step"] += 1
                    update = _adam_update(
                        parameter.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )

                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.reshape(parameter.shape), alpha=-group["lr"])

        return loss


def create_muon_optimizer(model, lr, weight_decay, momentum=0.95):
    """Use Muon for transformer matrix weights and AdamW for the remaining weights."""
    muon_parameters = []
    adam_parameters = []

    for name, parameter in model.named_parameters():
        # Restrict Muon to matrix weights inside Transformer-XL blocks.
        is_hidden_matrix = name.startswith("layers.") and parameter.ndim == 2
        (muon_parameters if is_hidden_matrix else adam_parameters).append(parameter)

    if not muon_parameters:
        raise ValueError("Muon requires at least one eligible matrix parameter.")
    if not adam_parameters:
        raise ValueError("Muon requires at least one parameter for its AdamW group.")

    return SingleDeviceMuonWithAuxAdam(
        [
            {
                "params": muon_parameters,
                "use_muon": True,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
            },
            {
                "params": adam_parameters,
                "use_muon": False,
                "lr": lr,
                "betas": (0.9, 0.95),
                "eps": 1e-10,
                "weight_decay": weight_decay,
            },
        ]
    )
