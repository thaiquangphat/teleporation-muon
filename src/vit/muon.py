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
    update = gradient.float()
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
        # Set by create_muon_optimizer. Diagnostics are opt-in and are
        # collected inside the real update to avoid replaying Newton--Schulz.
        self.diagnostics_enabled = False
        self.last_step_diagnostics = {}
        self._diagnostic_parameter_names = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if self.diagnostics_enabled:
            self.last_step_diagnostics = {}
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                if group["use_muon"]:
                    diagnostic_values = None
                    if self.diagnostics_enabled:
                        beta = float(group["momentum"])
                        gradient = parameter.grad.detach()
                        previous = state.get("momentum_buffer")
                        if torch.is_tensor(previous):
                            previous_for_metrics = previous.detach()
                        else:
                            previous_for_metrics = torch.zeros_like(gradient)
                        next_momentum = previous_for_metrics.lerp(gradient, 1.0 - beta)
                        raw_direction = gradient.lerp(next_momentum, beta)
                        diagnostic_values = {
                            "gradient_frobenius": float(gradient.float().norm().item()),
                            "momentum_before_frobenius": float(previous_for_metrics.float().norm().item()),
                            "momentum_gradient_cosine": float(torch.nn.functional.cosine_similarity(
                                previous_for_metrics.float().reshape(1, -1),
                                gradient.float().reshape(1, -1), dim=1, eps=1e-12
                            ).item()),
                            "raw_nesterov_direction_frobenius": float(raw_direction.float().norm().item()),
                        }
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    update = _muon_update(
                        parameter.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                    )
                    if diagnostic_values is not None:
                        update_for_metrics = update.reshape(parameter.shape).float()
                        parameter_float = parameter.detach().float()
                        gradient_float = parameter.grad.detach().float()
                        momentum_after = state["momentum_buffer"].detach().float()
                        lr = float(group["lr"])
                        weight_decay = float(group["weight_decay"])
                        actual_delta = -lr * (update_for_metrics + weight_decay * parameter_float)
                        diagnostic_values.update({
                            "momentum_after_frobenius": float(momentum_after.norm().item()),
                            "muon_direction_frobenius": float(update.float().norm().item()),
                            "gradient_muon_cosine": float(torch.nn.functional.cosine_similarity(
                                gradient_float.reshape(1, -1), update_for_metrics.reshape(1, -1), dim=1, eps=1e-12
                            ).item()),
                            "raw_nesterov_muon_direction_cosine": float(torch.nn.functional.cosine_similarity(
                                raw_direction.float().reshape(1, -1), update_for_metrics.reshape(1, -1),
                                dim=1, eps=1e-12
                            ).item()),
                            "weight_muon_cosine": float(torch.nn.functional.cosine_similarity(
                                parameter_float.reshape(1, -1), update_for_metrics.reshape(1, -1), dim=1, eps=1e-12
                            ).item()),
                            "actual_update_frobenius": float(actual_delta.norm().item()),
                            "relative_update_norm": float(
                                actual_delta.norm().item() / max(parameter_float.norm().item(), 1e-12)
                            ),
                        })
                        parameter_name = self._diagnostic_parameter_names.get(
                            id(parameter), f"parameter_{len(self.last_step_diagnostics)}"
                        )
                        self.last_step_diagnostics[parameter_name] = diagnostic_values
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


def create_muon_optimizer(model, lr, weight_decay, momentum=0.95, adam_lr=3e-4):
    """Use Muon for transformer matrix weights and AdamW for the remaining weights."""
    muon_parameters = []
    adam_parameters = []
    adam_only_name_parts = ("embedding", "classifier", "pooler")

    for name, parameter in model.named_parameters():
        is_hidden_matrix = (
            parameter.ndim >= 2
            and not any(part in name.lower() for part in adam_only_name_parts)
        )
        (muon_parameters if is_hidden_matrix else adam_parameters).append(parameter)

    if not muon_parameters:
        raise ValueError("Muon requires at least one eligible matrix parameter.")
    if not adam_parameters:
        raise ValueError("Muon requires at least one parameter for its AdamW group.")

    optimizer = SingleDeviceMuonWithAuxAdam(
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
                "lr": adam_lr,
                "betas": (0.9, 0.95),
                "eps": 1e-10,
                "weight_decay": weight_decay,
            },
        ]
    )
    optimizer._diagnostic_parameter_names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    return optimizer
