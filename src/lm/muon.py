# Copyright (c) 2024 Keller Jordan
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Single-device Muon optimizer with optional Nesterov momentum.

The ordinary Muon variant applies Newton--Schulz to the EMA momentum buffer
B_t. When ``nesterov=True``, it instead applies Newton--Schulz to the
Nesterov-corrected direction V_t = (1-beta)G_t + beta B_t.

This switch is deliberately opt-in: the plain and naive LM baselines use
ordinary Muon, while the Nesterov CSD experiment enables the Nesterov variant.
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


def _muon_update(gradient, momentum_buffer, beta, nesterov=False):
    """Return the Muon update direction before applying the learning rate."""
    momentum_buffer.lerp_(gradient, 1 - beta)
    if nesterov:
        # Matches the repository's CSD-Nesterov definition exactly:
        # V_t = (1-beta)G_t + beta B_t.
        update = gradient.lerp(momentum_buffer, beta)
    else:
        # Ordinary Muon uses the EMA momentum B_t without look-ahead.
        update = momentum_buffer

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
                group.setdefault("nesterov", False)
                group.setdefault("weight_decay", 0)
                if not isinstance(group["nesterov"], bool):
                    raise TypeError("Muon group 'nesterov' must be a bool.")
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)
        super().__init__(param_groups, {})
        # Diagnostics are opt-in and are populated only on selected steps.
        self.diagnostics_enabled = False
        self.last_step_diagnostics = {}
        self.parameter_names = {}

    @staticmethod
    def _diagnostic_cosine(left, right):
        a = left.detach().float().reshape(-1)
        b = right.detach().float().reshape(-1)
        denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        cosine = torch.dot(a, b) / denominator.clamp_min(1e-12)
        return torch.where(denominator > 1e-12, cosine, torch.zeros_like(cosine))

    def _record_matrix_diagnostics(self, name, parameter, gradient, momentum, raw_direction, update, group):
        weight32 = parameter.detach().float()
        grad32 = gradient.detach().float()
        momentum32 = momentum.detach().float()
        raw32 = raw_direction.detach().float()
        update32 = update.detach().float().reshape_as(weight32)
        lr = float(group.get("lr", 0.0))
        weight_decay = float(group.get("weight_decay", 0.0))
        actual_delta = -lr * update32 - lr * weight_decay * weight32

        def enqueue(entry_name, w, g, m, raw, u, delta):
            wnorm = torch.linalg.vector_norm(w)
            gnorm = torch.linalg.vector_norm(g)
            mnorm = torch.linalg.vector_norm(m)
            rawnorm = torch.linalg.vector_norm(raw)
            unorm = torch.linalg.vector_norm(u)
            delta_norm = torch.linalg.vector_norm(delta)
            keys = (
                "weight_frobenius", "gradient_frobenius", "momentum_frobenius",
                "raw_direction_frobenius", "muon_direction_frobenius",
                "actual_update_frobenius", "update_to_weight_ratio",
                "gradient_to_weight_ratio", "weight_muon_cosine",
                "gradient_muon_cosine", "momentum_muon_cosine",
                "raw_direction_muon_cosine", "gradient_momentum_cosine",
            )
            values = (
                wnorm, gnorm, mnorm, rawnorm, unorm, delta_norm,
                delta_norm / wnorm.clamp_min(1e-12),
                gnorm / wnorm.clamp_min(1e-12),
                self._diagnostic_cosine(w, u),
                self._diagnostic_cosine(g, u),
                self._diagnostic_cosine(m, u),
                self._diagnostic_cosine(raw, u),
                self._diagnostic_cosine(g, m),
            )
            self._pending_step_diagnostics.append((entry_name, keys, values))

        enqueue(name, weight32, grad32, momentum32, raw32, update32, actual_delta)
        # Muon acts on the fused GPT-2 c_attn matrix. These blockwise
        # diagnostics expose its Q/K/V components without changing that update.
        if name.endswith(".attn.c_attn.weight") and weight32.ndim == 2 and weight32.shape[0] % 3 == 0:
            hidden = weight32.shape[0] // 3
            for label, rows in (("Q", slice(0, hidden)), ("K", slice(hidden, 2 * hidden)), ("V", slice(2 * hidden, 3 * hidden))):
                enqueue(
                    f"{name}/{label}", weight32[rows], grad32[rows], momentum32[rows],
                    raw32[rows], update32[rows], actual_delta[rows],
                )

    def _flush_step_diagnostics(self):
        if not self._pending_step_diagnostics:
            self.last_step_diagnostics = {}
            return
        flattened = [value for _name, _keys, values in self._pending_step_diagnostics for value in values]
        # One device-to-host synchronization for all per-matrix diagnostics.
        numbers = torch.stack(flattened).detach().cpu().tolist()
        cursor = 0
        result = {}
        for name, keys, values in self._pending_step_diagnostics:
            count = len(keys)
            result[name] = {key: float(value) for key, value in zip(keys, numbers[cursor:cursor + count])}
            cursor += count
        self.last_step_diagnostics = result
        self._pending_step_diagnostics = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.last_step_diagnostics = {}
        self._pending_step_diagnostics = []

        parameter_index = 0
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
                        nesterov=group["nesterov"],
                    )
                    if self.diagnostics_enabled and parameter.ndim >= 2:
                        beta = float(group["momentum"])
                        momentum = state["momentum_buffer"]
                        raw_direction = (
                            parameter.grad.lerp(momentum, beta)
                            if group.get("nesterov", False) else momentum
                        )
                        name = self.parameter_names.get(id(parameter), f"parameter_{parameter_index}")
                        self._record_matrix_diagnostics(
                            name, parameter, parameter.grad, momentum,
                            raw_direction, update, group,
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
                parameter_index += 1

        if self.diagnostics_enabled:
            self._flush_step_diagnostics()
        return loss


def create_muon_optimizer(
    model, lr, weight_decay, momentum=0.95, adam_lr=3e-4, nesterov=False
):
    """Use Muon for transformer matrix weights and AdamW for other weights."""
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

    return SingleDeviceMuonWithAuxAdam(
        [
            {
                "params": muon_parameters,
                "use_muon": True,
                "lr": lr,
                "momentum": momentum,
                "nesterov": bool(nesterov),
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
