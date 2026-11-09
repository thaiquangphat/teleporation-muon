"""Research diagnostics for ViT + Muon teleportation experiments.

The helpers in this module are observational: they detach tensors and never
modify model parameters, gradients, optimizer state, or the RNG state.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn


def _matrix_view(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    value = tensor.detach().float()
    if value.ndim == 2:
        return value
    if value.ndim >= 3:
        return value.flatten(1)
    return None


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().float().item())


def cosine_similarity(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    a = left.detach().float().reshape(-1)
    b = right.detach().float().reshape(-1)
    if a.numel() != b.numel():
        return float("nan")
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if not torch.isfinite(denominator) or denominator.item() <= eps:
        return 0.0
    return _safe_float(torch.dot(a, b) / denominator.clamp_min(eps))


def _spectral_norm_power_iteration(matrix: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Deterministic spectral-norm estimate without consuming RNG state."""
    rows, columns = matrix.shape
    vector = torch.arange(1, columns + 1, device=matrix.device, dtype=matrix.dtype)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-30)
    for _ in range(iterations):
        left = matrix @ vector
        left = left / torch.linalg.vector_norm(left).clamp_min(1e-30)
        vector = matrix.mT @ left
        vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-30)
    return torch.linalg.vector_norm(matrix @ vector)


def matrix_statistics(
    matrix: torch.Tensor,
    prefix: str,
    include_spectrum: bool = False,
) -> Dict[str, float]:
    """Return matrix geometry; exact rank/spectrum is computed on scheduled steps."""
    value = _matrix_view(matrix)
    if value is None:
        return {}
    value = value.contiguous()
    fro = torch.linalg.vector_norm(value)
    result = {f"{prefix}/frobenius": _safe_float(fro)}

    if not include_spectrum:
        spectral_estimate = _spectral_norm_power_iteration(value)
        result.update({
            f"{prefix}/spectral_estimate": _safe_float(spectral_estimate),
            f"{prefix}/stable_rank_estimate": _safe_float(
                fro.square() / spectral_estimate.square().clamp_min(1e-30)
            ),
        })
        return result

    singular_values = torch.linalg.svdvals(value)
    if singular_values.numel() == 0:
        return result
    smax = singular_values[0]
    smin = singular_values[-1]
    eps = torch.finfo(singular_values.dtype).eps
    tolerance = eps * max(value.shape) * smax
    numerical_rank = int((singular_values > tolerance).sum().item())
    nonzero = singular_values[singular_values > tolerance]
    nuclear = singular_values.sum()

    p_l1 = singular_values / singular_values.sum().clamp_min(1e-30)
    entropy_l1 = -(p_l1 * p_l1.clamp_min(1e-30).log()).sum()
    effective_rank_l1 = torch.exp(entropy_l1)

    energy = singular_values.square()
    p_energy = energy / energy.sum().clamp_min(1e-30)
    entropy_energy = -(p_energy * p_energy.clamp_min(1e-30).log()).sum()
    effective_rank_energy = torch.exp(entropy_energy)

    full_rank = numerical_rank == min(value.shape)
    condition_number = (smax / smin.clamp_min(1e-30)) if full_rank else None
    result.update({
        f"{prefix}/spectral": _safe_float(smax),
        f"{prefix}/nuclear": _safe_float(nuclear),
        f"{prefix}/stable_rank": _safe_float(fro.square() / smax.square().clamp_min(1e-30)),
        f"{prefix}/effective_rank_singular_entropy": _safe_float(effective_rank_l1),
        f"{prefix}/effective_rank_energy_entropy": _safe_float(effective_rank_energy),
        f"{prefix}/numerical_rank": float(numerical_rank),
        f"{prefix}/smallest_singular_value": _safe_float(smin),
        f"{prefix}/rank_deficient": float(not full_rank),
        f"{prefix}/singular_value_mean": _safe_float(singular_values.mean()),
        f"{prefix}/singular_value_std": _safe_float(singular_values.std(unbiased=False)),
        f"{prefix}/singular_value_top1": _safe_float(singular_values[0]),
        f"{prefix}/singular_value_top5_mean": _safe_float(
            singular_values[: min(5, singular_values.numel())].mean()
        ),
    })
    if condition_number is not None:
        result[f"{prefix}/condition_number"] = _safe_float(condition_number)
    return result


def _get_vit_layers(model: nn.Module):
    vit = getattr(model, "vit", None)
    encoder = getattr(vit, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        return []
    return list(layers)


def collect_weight_tensors(model: nn.Module, clone: bool = False) -> Dict[str, torch.Tensor]:
    """Collect Q/K/V/O weights and their parameter-space compositions.

    Conventions:
      QK_bilinear = W_Q^T W_K, so x W_Q^T W_K x^T is the query-key
      bilinear form for an input row vector x.
      VO_composition = W_O W_V, matching the row-vector composition
      x W_V^T W_O^T = x (W_O W_V)^T.
    """
    tensors: Dict[str, torch.Tensor] = {}
    for layer_index, block in enumerate(_get_vit_layers(model)):
        attention = getattr(block, "attention", None)
        self_attention = getattr(attention, "attention", None)
        output_module = getattr(attention, "output", None)
        projections = {
            "Q": getattr(self_attention, "query", None),
            "K": getattr(self_attention, "key", None),
            "V": getattr(self_attention, "value", None),
            "O": getattr(output_module, "dense", None),
        }
        prefix = f"layers/{layer_index}"
        for label, module in projections.items():
            weight = getattr(module, "weight", None)
            if torch.is_tensor(weight):
                detached = weight.detach()
                tensors[f"{prefix}/weights/{label}"] = detached.clone() if clone else detached
        q = projections["Q"]
        k = projections["K"]
        v = projections["V"]
        o = projections["O"]
        if all(module is not None and torch.is_tensor(getattr(module, "weight", None))
               for module in (q, k)):
            q_weight = q.weight.detach().float()
            k_weight = k.weight.detach().float()
            if q_weight.shape[0] == k_weight.shape[0]:
                tensors[f"{prefix}/weights/QK_bilinear"] = q_weight.mT @ k_weight
        if all(module is not None and torch.is_tensor(getattr(module, "weight", None))
               for module in (v, o)):
            v_weight = v.weight.detach().float()
            o_weight = o.weight.detach().float()
            if o_weight.shape[1] == v_weight.shape[0]:
                tensors[f"{prefix}/weights/VO_composition"] = o_weight @ v_weight
    return tensors


def collect_weight_geometry(
    model: nn.Module,
    include_spectrum: bool = False,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, tensor in collect_weight_tensors(model).items():
        metric_name = name.replace("/weights/", "/")
        metrics.update(matrix_statistics(tensor, f"weights/{metric_name}", include_spectrum))
    return metrics


def collect_attention_gradient_geometry(
    model: nn.Module,
    include_spectrum: bool = False,
) -> Dict[str, float]:
    """Log gradient geometry for Q/K/V/O projection weights."""
    metrics: Dict[str, float] = {}
    for layer_index, block in enumerate(_get_vit_layers(model)):
        attention = getattr(block, "attention", None)
        self_attention = getattr(attention, "attention", None)
        output_module = getattr(attention, "output", None)
        modules = {
            "Q": getattr(self_attention, "query", None),
            "K": getattr(self_attention, "key", None),
            "V": getattr(self_attention, "value", None),
            "O": getattr(output_module, "dense", None),
        }
        for label, module in modules.items():
            weight = getattr(module, "weight", None)
            gradient = getattr(weight, "grad", None)
            if not torch.is_tensor(weight) or not torch.is_tensor(gradient):
                continue
            prefix = f"gradients/layers/{layer_index}/{label}"
            metrics.update(matrix_statistics(gradient, prefix, include_spectrum))
            metrics[f"{prefix}/to_weight_frobenius_ratio"] = (
                _safe_float(torch.linalg.vector_norm(gradient.float()))
                / max(_safe_float(torch.linalg.vector_norm(weight.detach().float())), 1e-12)
            )
    return metrics


class AttentionActivationRecorder:
    """Captures one selected forward pass of Q/K/V, post-PE QK and VO outputs.

    Hooks are installed once. They retain no activations unless ``enabled`` is
    true. For the repository's RoPE implementation, Q/K are replaced by the
    post-rotary tensors exposed by ViTRoPESelfAttention diagnostics.
    """

    def __init__(self, model: nn.Module):
        self.enabled = False
        self.layers: Dict[int, Dict[str, torch.Tensor]] = {}
        self.handles = []
        self.attention_modules = []
        self.num_heads = int(getattr(getattr(model, "config", None), "num_attention_heads", 1))
        config = getattr(model, "config", None)
        if config is None and hasattr(model, "vit"):
            config = getattr(model.vit, "config", None)
        self.num_heads = int(getattr(config, "num_attention_heads", self.num_heads))

        for layer_index, block in enumerate(_get_vit_layers(model)):
            attention = getattr(block, "attention", None)
            self_attention = getattr(attention, "attention", None)
            output_module = getattr(attention, "output", None)
            if self_attention is None:
                continue
            self.attention_modules.append(self_attention)
            self.layers[layer_index] = {}

            projection_labels = {"query": "Q", "key": "K", "value": "V"}
            for label, short_label in projection_labels.items():
                module = getattr(self_attention, label, None)
                if module is None:
                    continue
                self.handles.append(
                    module.register_forward_hook(self._projection_hook(layer_index, short_label))
                )

            self.handles.append(
                self_attention.register_forward_hook(self._attention_hook(layer_index))
            )
            dense = getattr(output_module, "dense", None)
            if dense is not None:
                self.handles.append(
                    dense.register_forward_pre_hook(self._output_input_hook(layer_index))
                )
                self.handles.append(
                    dense.register_forward_hook(self._output_hook(layer_index))
                )

    def _projection_hook(self, layer_index: int, label: str):
        def hook(_module, _inputs, output):
            if self.enabled and torch.is_tensor(output):
                self.layers[layer_index][label] = output.detach()
        return hook

    def _attention_hook(self, layer_index: int):
        def hook(module, _inputs, output):
            if not self.enabled:
                return
            values = self.layers[layer_index]
            stored = getattr(module, "last_diagnostics", None)
            if isinstance(stored, dict):
                for key, value in stored.items():
                    if torch.is_tensor(value):
                        values[key.upper()] = value.detach()
            if "Q_POST_PE" in values:
                values.setdefault("Q_PRE_PE", values.get("Q"))
                values["Q"] = values["Q_POST_PE"]
            if "K_POST_PE" in values:
                values.setdefault("K_PRE_PE", values.get("K"))
                values["K"] = values["K_POST_PE"]
            if isinstance(output, (tuple, list)):
                if len(output) > 1 and torch.is_tensor(output[1]):
                    values["ATTENTION_PROBS"] = output[1].detach()
                if len(output) > 0 and torch.is_tensor(output[0]):
                    values.setdefault("CONTEXT", output[0].detach())

            # For the standard learned-position ViT, Q/K projection outputs
            # already include the position-augmented token representation.
            # The custom RoPE path supplies the exact post-rotation product.
            if "QK_LOGITS" not in values:
                q, k = values.get("Q"), values.get("K")
                if torch.is_tensor(q) and torch.is_tensor(k):
                    if q.ndim == 3:
                        batch, sequence, hidden = q.shape
                        heads = max(1, self.num_heads)
                        if hidden % heads == 0:
                            q = q.reshape(batch, sequence, heads, hidden // heads).permute(0, 2, 1, 3)
                            k = k.reshape(batch, sequence, heads, hidden // heads).permute(0, 2, 1, 3)
                    if q.ndim == 4 and k.ndim == 4:
                        values["QK_LOGITS"] = (
                            torch.matmul(q.float(), k.float().transpose(-1, -2))
                            * (q.shape[-1] ** -0.5)
                        ).detach()
            if "ATTENTION_PROBS" not in values and torch.is_tensor(values.get("QK_LOGITS")):
                # The project ViT uses zero attention dropout and no causal or
                # padding mask, so this reconstructs probabilities when an
                # SDPA/flash backend does not return them explicitly.
                values["ATTENTION_PROBS"] = torch.softmax(values["QK_LOGITS"].float(), dim=-1).detach()
        return hook

    def _output_input_hook(self, layer_index: int):
        def hook(_module, inputs):
            if self.enabled and inputs and torch.is_tensor(inputs[0]):
                self.layers[layer_index]["O_INPUT"] = inputs[0].detach()
        return hook

    def _output_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            if self.enabled and torch.is_tensor(output):
                self.layers[layer_index]["O_OUTPUT"] = output.detach()
        return hook

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        for module in self.attention_modules:
            if hasattr(module, "diagnostics_enabled"):
                module.diagnostics_enabled = self.enabled
            if not self.enabled and hasattr(module, "last_diagnostics"):
                module.last_diagnostics = None

    def get_tensors(self, cpu: bool = True, max_batch: Optional[int] = 1) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        for layer_index, values in self.layers.items():
            q = values.get("Q")
            k = values.get("K")
            v = values.get("V")
            q_post = values.get("Q_POST_PE")
            k_post = values.get("K_POST_PE")
            if q_post is not None:
                q = q_post
            if k_post is not None:
                k = k_post
            # Standard Hugging Face ViT hooks see [B, S, hidden]; normalize
            # them to the same [B, heads, S, head_dim] layout as post-RoPE.
            if torch.is_tensor(q) and q.ndim == 3:
                batch, sequence, hidden = q.shape
                heads = max(1, self.num_heads)
                if hidden % heads == 0:
                    q = q.reshape(batch, sequence, heads, hidden // heads).permute(0, 2, 1, 3)
            if torch.is_tensor(k) and k.ndim == 3:
                batch, sequence, hidden = k.shape
                heads = max(1, self.num_heads)
                if hidden % heads == 0:
                    k = k.reshape(batch, sequence, heads, hidden // heads).permute(0, 2, 1, 3)
            if torch.is_tensor(v) and v.ndim == 3:
                batch, sequence, hidden = v.shape
                heads = max(1, self.num_heads)
                if hidden % heads == 0:
                    v = v.reshape(batch, sequence, heads, hidden // heads).permute(0, 2, 1, 3)

            prefix = f"layers/{layer_index}/activations"
            # The custom RoPE model exposes both the raw projection output and
            # the post-rotary tensor. Standard learned-PE ViT exposes only the
            # latter-stage Q/K at this point because PE was added to hidden states.
            for label in ("Q_PRE_PE", "K_PRE_PE"):
                raw_tensor = values.get(label)
                if torch.is_tensor(raw_tensor):
                    if max_batch is not None and raw_tensor.ndim >= 3:
                        raw_tensor = raw_tensor[:max(1, int(max_batch))]
                    tensors[f"{prefix}/{label}"] = raw_tensor.detach().cpu() if cpu else raw_tensor.detach()
            for label, tensor in (("Q", q), ("K", k), ("V", v)):
                if torch.is_tensor(tensor):
                    if max_batch is not None and tensor.ndim >= 3:
                        tensor = tensor[:max(1, int(max_batch))]
                    tensors[f"{prefix}/{label}"] = tensor.detach().cpu() if cpu else tensor.detach()

            raw_q = values.get("Q_PRE_PE")
            raw_k = values.get("K_PRE_PE")
            if torch.is_tensor(raw_q) and torch.is_tensor(raw_k):
                if raw_q.ndim == 3:
                    batch, sequence, hidden = raw_q.shape
                    heads = max(1, self.num_heads)
                    if hidden % heads == 0:
                        raw_q = raw_q.reshape(batch, sequence, heads, hidden // heads).permute(0, 2, 1, 3)
                        raw_k = raw_k.reshape(batch, sequence, heads, hidden // heads).permute(0, 2, 1, 3)
                if raw_q.ndim == 4 and raw_k.ndim == 4:
                    qk_before = torch.matmul(raw_q.float(), raw_k.float().transpose(-1, -2))
                    qk_before = qk_before * (raw_q.shape[-1] ** -0.5)
                    if max_batch is not None:
                        qk_before = qk_before[:max(1, int(max_batch))]
                    tensors[f"{prefix}/QK_before_PE"] = qk_before.detach().cpu() if cpu else qk_before.detach()

            if torch.is_tensor(q) and torch.is_tensor(k):
                qk = values.get("QK_LOGITS")
                if qk is None and q.ndim == 4 and k.ndim == 4:
                    qk = torch.matmul(q.float(), k.float().transpose(-1, -2))
                    qk = qk * (q.shape[-1] ** -0.5)
                if torch.is_tensor(qk):
                    if max_batch is not None and qk.ndim >= 3:
                        qk = qk[:max(1, int(max_batch))]
                    tensors[f"{prefix}/QK_after_PE"] = qk.detach().cpu() if cpu else qk.detach()

            aliases = {
                "ATTENTION_PROBS": "attention_probabilities",
                "O_INPUT": "V_attention_context_before_O",
                "O_OUTPUT": "VO_after_attention",
                "CONTEXT": "attention_context",
            }
            for source, target in aliases.items():
                tensor = values.get(source)
                if torch.is_tensor(tensor):
                    if max_batch is not None and tensor.ndim >= 3:
                        tensor = tensor[:max(1, int(max_batch))]
                    tensors[f"{prefix}/{target}"] = tensor.detach().cpu() if cpu else tensor.detach()
        return tensors

    def get_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for layer_index, values in self.layers.items():
            prefix = f"attention/layers/{layer_index}"
            for key in ("Q_PRE_PE", "K_PRE_PE", "Q", "K", "V", "O_INPUT", "O_OUTPUT", "QK_LOGITS"):
                tensor = values.get(key)
                if not torch.is_tensor(tensor):
                    continue
                value = tensor.detach().float()
                metrics[f"{prefix}/{key.lower()}_mean"] = _safe_float(value.mean())
                metrics[f"{prefix}/{key.lower()}_std"] = _safe_float(value.std(unbiased=False))
                metrics[f"{prefix}/{key.lower()}_rms"] = _safe_float(value.square().mean().sqrt())
                metrics[f"{prefix}/{key.lower()}_max_abs"] = _safe_float(value.abs().max())
                metrics[f"{prefix}/{key.lower()}_frobenius"] = _safe_float(torch.linalg.vector_norm(value))

            probs = values.get("ATTENTION_PROBS")
            if torch.is_tensor(probs):
                probabilities = probs.detach().float().clamp_min(1e-12)
                entropy = -(probabilities * probabilities.log()).sum(dim=-1)
                sequence_length = probabilities.shape[-1]
                metrics[f"{prefix}/attention_entropy"] = _safe_float(entropy.mean())
                metrics[f"{prefix}/attention_entropy_normalized"] = _safe_float(
                    entropy.mean() / math.log(max(sequence_length, 2))
                )
                metrics[f"{prefix}/attention_max_probability"] = _safe_float(probs.detach().float().max())
                if probabilities.ndim == 4:
                    per_head_entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean(dim=(0, 2))
                    per_head_max = probs.detach().float().amax(dim=-1).mean(dim=(0, 2))
                    for head_index in range(per_head_entropy.numel()):
                        head_prefix = f"{prefix}/heads/{head_index}"
                        metrics[f"{head_prefix}/attention_entropy"] = _safe_float(
                            per_head_entropy[head_index]
                        )
                        metrics[f"{head_prefix}/attention_entropy_normalized"] = _safe_float(
                            per_head_entropy[head_index] / math.log(max(sequence_length, 2))
                        )
                        metrics[f"{head_prefix}/attention_mean_max_probability"] = _safe_float(
                            per_head_max[head_index]
                        )
        return metrics

    def clear(self) -> None:
        for values in self.layers.values():
            values.clear()
        for module in self.attention_modules:
            if hasattr(module, "last_diagnostics"):
                module.last_diagnostics = None

    def close(self) -> None:
        self.set_enabled(False)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_csd_diagnostics(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    use_nesterov_score: bool,
) -> Dict[str, float]:
    """Compute per-matrix CSD dual norms and their normalized contributions.

    The CSD score is the L2 aggregation of nuclear norms. The L1-normalized
    shares are also reported; unlike the L2 coefficients, these yield a useful
    effective number of contributing matrices.
    """
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    entries = []
    for group in optimizer.param_groups:
        if not bool(group.get("use_muon", False)):
            continue
        beta = float(group.get("momentum", 0.95))
        for parameter in group["params"]:
            if parameter.ndim < 2 or parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            previous = optimizer.state.get(parameter, {}).get("momentum_buffer")
            if torch.is_tensor(previous):
                previous = previous.detach().float()
            else:
                previous = torch.zeros_like(gradient)
            if use_nesterov_score:
                direction = previous.mul(beta * beta).add(gradient, alpha=1.0 - beta * beta)
            else:
                direction = previous.mul(beta).add(gradient, alpha=1.0 - beta)
            matrix = _matrix_view(direction)
            nuclear = _safe_float(torch.linalg.matrix_norm(matrix, ord="nuc"))
            entries.append((names.get(id(parameter), f"parameter_{len(entries)}"), nuclear))

    if not entries:
        return {}
    values = torch.tensor([item[1] for item in entries], dtype=torch.float64)
    denominator_l2 = float(torch.linalg.vector_norm(values).item())
    denominator_l1 = float(values.sum().item())
    metrics: Dict[str, float] = {
        "csd/objective": denominator_l2,
        "csd/sum_nuclear_norms": denominator_l1,
        "csd/matrix_count": float(len(entries)),
    }
    if denominator_l1 > 0:
        shares = values / denominator_l1
        metrics["csd/effective_matrix_count_l1"] = float(1.0 / shares.square().sum().item())
        metrics["csd/max_l1_share"] = float(shares.max().item())
        metrics["csd/share_entropy"] = float(
            -(shares * shares.clamp_min(1e-30).log()).sum().item()
        )
    for name, nuclear in entries:
        safe_name = name.replace(".", "/")
        metrics[f"csd/matrices/{safe_name}/nuclear_norm"] = nuclear
        metrics[f"csd/matrices/{safe_name}/coefficient_l2"] = (
            nuclear / denominator_l2 if denominator_l2 > 0 else 0.0
        )
        metrics[f"csd/matrices/{safe_name}/share_l1"] = (
            nuclear / denominator_l1 if denominator_l1 > 0 else 0.0
        )
    return metrics


def optimizer_diagnostic_metrics(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """Flatten the most recent scalar diagnostics emitted by the Muon step."""
    result: Dict[str, float] = {}
    for parameter_name, values in getattr(optimizer, "last_step_diagnostics", {}).items():
        safe_name = parameter_name.replace(".", "/")
        for metric_name, metric_value in values.items():
            if metric_value is not None and math.isfinite(float(metric_value)):
                result[f"muon/matrices/{safe_name}/{metric_name}"] = float(metric_value)
    return result


def build_tensor_snapshot(
    model: nn.Module,
    activation_recorder: AttentionActivationRecorder,
) -> Dict[str, torch.Tensor]:
    # Clone before the optimizer step: .detach() alone aliases live parameters.
    snapshot = collect_weight_tensors(model, clone=True)
    # Save one representative image per layer for activation tensors; full
    # parameter matrices and parameter-space products remain untruncated.
    snapshot.update(activation_recorder.get_tensors(cpu=True, max_batch=1))
    return {name: tensor.detach().cpu().contiguous().clone() for name, tensor in snapshot.items()}


def log_tensor_visuals(
    run: Any,
    weight_tensors: Mapping[str, torch.Tensor],
    activation_tensors: Mapping[str, torch.Tensor],
    step: int,
    max_image_layers: int = 3,
) -> None:
    """Log distributions for all weight matrices and heatmaps for selected layers."""
    import wandb

    payload: Dict[str, Any] = {}
    for name, tensor in weight_tensors.items():
        flattened = tensor.detach().float().reshape(-1).cpu()
        if flattened.numel():
            payload[f"tensor_histograms/{name}"] = wandb.Histogram(flattened.numpy())

    layer_ids = sorted({
        int(name.split("/")[1])
        for name in list(weight_tensors.keys()) + list(activation_tensors.keys())
        if name.startswith("layers/") and len(name.split("/")) > 1 and name.split("/")[1].isdigit()
    })
    if layer_ids:
        selected = sorted(set([layer_ids[0], layer_ids[len(layer_ids) // 2], layer_ids[-1]]))
        selected = selected[: max(1, int(max_image_layers))]
        for layer_id in selected:
            prefix = f"layers/{layer_id}"
            for name, tensor in weight_tensors.items():
                if not name.startswith(prefix + "/weights/"):
                    continue
                value = tensor.detach().float().cpu()
                if value.ndim == 2:
                    payload[f"tensor_images/{name}"] = wandb.Image(value.numpy())
            for name, tensor in activation_tensors.items():
                if not name.startswith(prefix + "/activations/"):
                    continue
                value = tensor.detach().float().cpu()
                if value.ndim >= 4:
                    value = value[0, 0]
                elif value.ndim == 3:
                    value = value[0]
                if value.ndim == 2:
                    payload[f"tensor_images/{name}"] = wandb.Image(value.numpy())
    if payload:
        payload["diagnostics/tensor_snapshot_step"] = int(step)
        run.log(payload)


def log_tensor_artifact(
    run: Any,
    snapshot: Mapping[str, torch.Tensor],
    save_dir: str,
    step: int,
    epoch: int,
) -> str:
    """Persist exact tensors locally and upload them as a W&B artifact."""
    import wandb

    artifact_dir = os.path.join(save_dir, "diagnostics")
    os.makedirs(artifact_dir, exist_ok=True)
    filename = f"matrix_tensors_epoch_{int(epoch):04d}_step_{int(step):08d}.pt"
    path = os.path.join(artifact_dir, filename)
    torch.save({name: tensor.detach().cpu() for name, tensor in snapshot.items()}, path)
    artifact = wandb.Artifact(
        name=f"matrix-tensors-{run.id}-e{int(epoch)}-s{int(step)}",
        type="matrix-diagnostics",
        metadata={
            "epoch": int(epoch),
            "global_step": int(step),
            "tensor_count": len(snapshot),
            "activation_batch_samples": 1,
        },
    )
    artifact.add_file(path, name=filename)
    run.log_artifact(artifact)
    return path
