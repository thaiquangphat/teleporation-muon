"""Research diagnostics for GPT-2 / Muon / CSD teleportation experiments.

All collectors are observational: they detach tensors and do not modify model,
gradient, optimizer, or RNG state. Expensive SVDs and tensor artifacts are only
requested at explicitly configured diagnostic intervals.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn


def _safe_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().item())
    return float(value)


def cosine_similarity(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    a = left.detach().float().reshape(-1)
    b = right.detach().float().reshape(-1)
    if a.numel() != b.numel():
        return float("nan")
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if not torch.isfinite(denom) or denom.item() <= eps:
        return 0.0
    return _safe_float(torch.dot(a, b) / denom.clamp_min(eps))


def _matrix_view(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    value = tensor.detach().float()
    if value.ndim == 2:
        return value
    if value.ndim >= 3:
        return value.flatten(1)
    return None


def _spectral_norm_power_iteration(matrix: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """Deterministic spectral estimate; does not consume RNG state."""
    rows, columns = matrix.shape
    vector = torch.arange(1, columns + 1, device=matrix.device, dtype=matrix.dtype)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-30)
    for _ in range(max(1, int(iterations))):
        left = matrix @ vector
        left = left / torch.linalg.vector_norm(left).clamp_min(1e-30)
        vector = matrix.mT @ left
        vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-30)
    return torch.linalg.vector_norm(matrix @ vector)


def matrix_statistics(
    tensor: torch.Tensor,
    prefix: str,
    include_spectrum: bool = False,
) -> Dict[str, float]:
    value = _matrix_view(tensor)
    if value is None:
        return {}
    value = value.contiguous()
    fro = torch.linalg.vector_norm(value)
    result = {f"{prefix}/frobenius": _safe_float(fro)}
    if not include_spectrum:
        spectral = _spectral_norm_power_iteration(value)
        result[f"{prefix}/spectral_estimate"] = _safe_float(spectral)
        result[f"{prefix}/stable_rank_estimate"] = _safe_float(
            fro.square() / spectral.square().clamp_min(1e-30)
        )
        return result

    singular_values = torch.linalg.svdvals(value)
    if singular_values.numel() == 0:
        return result
    smax, smin = singular_values[0], singular_values[-1]
    tolerance = torch.finfo(singular_values.dtype).eps * max(value.shape) * smax
    numerical_rank = int((singular_values > tolerance).sum().item())
    p_l1 = singular_values / singular_values.sum().clamp_min(1e-30)
    entropy_l1 = -(p_l1 * p_l1.clamp_min(1e-30).log()).sum()
    energy = singular_values.square()
    p_energy = energy / energy.sum().clamp_min(1e-30)
    entropy_energy = -(p_energy * p_energy.clamp_min(1e-30).log()).sum()
    full_rank = numerical_rank == min(value.shape)
    result.update({
        f"{prefix}/spectral": _safe_float(smax),
        f"{prefix}/nuclear": _safe_float(singular_values.sum()),
        f"{prefix}/stable_rank": _safe_float(fro.square() / smax.square().clamp_min(1e-30)),
        f"{prefix}/effective_rank_singular_entropy": _safe_float(entropy_l1.exp()),
        f"{prefix}/effective_rank_energy_entropy": _safe_float(entropy_energy.exp()),
        f"{prefix}/numerical_rank": float(numerical_rank),
        f"{prefix}/smallest_singular_value": _safe_float(smin),
        f"{prefix}/rank_deficient": float(not full_rank),
        f"{prefix}/singular_value_mean": _safe_float(singular_values.mean()),
        f"{prefix}/singular_value_std": _safe_float(singular_values.std(unbiased=False)),
        f"{prefix}/singular_value_top1": _safe_float(smax),
        f"{prefix}/singular_value_top5_mean": _safe_float(singular_values[:5].mean()),
    })
    if full_rank:
        result[f"{prefix}/condition_number"] = _safe_float(smax / smin.clamp_min(1e-30))
    return result


def _lm_layers(model: nn.Module) -> list[nn.Module]:
    return list(getattr(getattr(model, "transformer", None), "h", []))


def diagnostic_layer_indices(model: nn.Module, count: int = 3) -> list[int]:
    n_layers = len(_lm_layers(model))
    if n_layers == 0:
        return []
    count = max(1, min(int(count), n_layers))
    if count == 1:
        return [(n_layers - 1) // 2]
    raw = [round(i * (n_layers - 1) / (count - 1)) for i in range(count)]
    return sorted(set(raw))


def collect_weight_tensors(
    model: nn.Module,
    layer_indices: Optional[Sequence[int]] = None,
    clone: bool = False,
) -> Dict[str, torch.Tensor]:
    """Collect fused-QKV splits, O/MLP matrices and effective QK/VO products.

    For row-vector activations, QK_bilinear=W_Q^T W_K. The value/output
    composition is W_O W_V, since x W_V^T W_O^T=x (W_O W_V)^T.
    """
    layers = _lm_layers(model)
    selected = set(range(len(layers))) if layer_indices is None else set(map(int, layer_indices))
    tensors: Dict[str, torch.Tensor] = {}
    for index, block in enumerate(layers):
        if index not in selected:
            continue
        hidden = int(model.config.n_embd)
        fused = block.attn.c_attn.weight.detach()
        q, k, v = fused.split(hidden, dim=0)
        matrices = {
            "Q": q, "K": k, "V": v,
            "O": block.attn.c_proj.weight.detach(),
            "MLP_IN": block.mlp.c_fc.weight.detach(),
            "MLP_OUT": block.mlp.c_proj.weight.detach(),
            "QK_bilinear": q.float().mT @ k.float(),
            "VO_composition": block.attn.c_proj.weight.detach().float() @ v.float(),
        }
        prefix = f"layers/{index}/weights"
        for name, tensor in matrices.items():
            item = tensor.detach()
            tensors[f"{prefix}/{name}"] = item.clone() if clone else item
    return tensors


def collect_weight_geometry(
    model: nn.Module,
    include_spectrum: bool = False,
    spectrum_layers: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    expensive_layers = set(spectrum_layers or [])
    for name, tensor in collect_weight_tensors(model).items():
        parts = name.split("/")
        layer_index = int(parts[1])
        include = include_spectrum and layer_index in expensive_layers
        metric_prefix = name.replace("/weights/", "/")
        metrics.update(matrix_statistics(tensor, metric_prefix, include_spectrum=include))
    return metrics


def collect_gradient_geometry(
    model: nn.Module,
    include_spectrum: bool = False,
    spectrum_layers: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    expensive_layers = set(spectrum_layers or [])
    for layer_index, block in enumerate(_lm_layers(model)):
        hidden = int(model.config.n_embd)
        modules = {
            "Q": (block.attn.c_attn.weight, slice(0, hidden)),
            "K": (block.attn.c_attn.weight, slice(hidden, 2 * hidden)),
            "V": (block.attn.c_attn.weight, slice(2 * hidden, 3 * hidden)),
            "O": (block.attn.c_proj.weight, slice(None)),
            "MLP_IN": (block.mlp.c_fc.weight, slice(None)),
            "MLP_OUT": (block.mlp.c_proj.weight, slice(None)),
        }
        for label, (weight, rows) in modules.items():
            grad = weight.grad
            if grad is None:
                continue
            current_weight = weight.detach()[rows]
            current_grad = grad.detach()[rows]
            prefix = f"gradients/layers/{layer_index}/{label}"
            metrics.update(matrix_statistics(
                current_grad, prefix,
                include_spectrum=include_spectrum and layer_index in expensive_layers,
            ))
            metrics[f"{prefix}/to_weight_frobenius_ratio"] = (
                _safe_float(torch.linalg.vector_norm(current_grad.float()))
                / max(_safe_float(torch.linalg.vector_norm(current_weight.float())), 1e-12)
            )
    return metrics


class GPT2ActivationRecorder:
    """Enables gated activation capture on GPT-2 attention and MLP modules."""
    def __init__(self, model: nn.Module, layer_indices: Optional[Sequence[int]] = None):
        self.enabled = False
        self.attention_modules = []
        self.mlp_modules = []
        n_layers = len(_lm_layers(model))
        self.layer_indices = set(range(n_layers)) if layer_indices is None else set(map(int, layer_indices))
        for block in _lm_layers(model):
            self.attention_modules.append(block.attn)
            self.mlp_modules.append(block.mlp)
        self.set_enabled(False)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        for index, module in enumerate(self.attention_modules):
            module.diagnostics_enabled = self.enabled and index in self.layer_indices
            if not module.diagnostics_enabled:
                module.last_diagnostics = None
        for index, module in enumerate(self.mlp_modules):
            module.diagnostics_enabled = self.enabled and index in self.layer_indices
            if not module.diagnostics_enabled:
                module.last_diagnostics = None

    def clear(self) -> None:
        for module in self.attention_modules + self.mlp_modules:
            module.last_diagnostics = None

    def _all_values(self) -> Dict[int, Dict[str, torch.Tensor]]:
        values: Dict[int, Dict[str, torch.Tensor]] = {}
        for index, module in enumerate(self.attention_modules):
            if isinstance(module.last_diagnostics, dict):
                values.setdefault(index, {}).update(module.last_diagnostics)
        for index, module in enumerate(self.mlp_modules):
            if isinstance(module.last_diagnostics, dict):
                values.setdefault(index, {}).update({f"MLP_{k}": v for k, v in module.last_diagnostics.items()})
        return values

    def get_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for layer_index, values in self._all_values().items():
            prefix = f"activations/layers/{layer_index}"
            for key, tensor in values.items():
                if not torch.is_tensor(tensor):
                    continue
                value = tensor.detach().float()
                key_prefix = f"{prefix}/{key.lower()}"
                metrics[f"{key_prefix}/mean"] = _safe_float(value.mean())
                metrics[f"{key_prefix}/std"] = _safe_float(value.std(unbiased=False))
                metrics[f"{key_prefix}/rms"] = _safe_float(value.square().mean().sqrt())
                metrics[f"{key_prefix}/max_abs"] = _safe_float(value.abs().max())
                metrics[f"{key_prefix}/frobenius"] = _safe_float(torch.linalg.vector_norm(value))

            probabilities = values.get("ATTENTION_PROBS")
            if torch.is_tensor(probabilities):
                probs = probabilities.detach().float().clamp_min(1e-12)
                entropy = -(probs * probs.log()).sum(dim=-1)
                seq_len = probs.shape[-1]
                metrics[f"{prefix}/attention_entropy"] = _safe_float(entropy.mean())
                metrics[f"{prefix}/attention_entropy_normalized"] = _safe_float(
                    entropy.mean() / math.log(max(seq_len, 2))
                )
                metrics[f"{prefix}/attention_max_probability"] = _safe_float(probs.max())
                if probs.ndim == 4:
                    by_head = entropy.mean(dim=(0, 2))
                    max_by_head = probs.max(dim=-1).values.mean(dim=(0, 2))
                    for head in range(by_head.numel()):
                        metrics[f"{prefix}/heads/{head}/attention_entropy"] = _safe_float(by_head[head])
                        metrics[f"{prefix}/heads/{head}/attention_entropy_normalized"] = _safe_float(
                            by_head[head] / math.log(max(seq_len, 2))
                        )
                        metrics[f"{prefix}/heads/{head}/mean_max_probability"] = _safe_float(max_by_head[head])
        return metrics

    def get_tensors(
        self,
        layer_indices: Optional[Sequence[int]] = None,
        max_batch: int = 1,
        max_sequence_length: int = 128,
        cpu: bool = True,
    ) -> Dict[str, torch.Tensor]:
        selected = None if layer_indices is None else set(map(int, layer_indices))
        tensors: Dict[str, torch.Tensor] = {}
        for layer_index, values in self._all_values().items():
            if selected is not None and layer_index not in selected:
                continue
            for name, tensor in values.items():
                if not torch.is_tensor(tensor):
                    continue
                value = tensor.detach()
                if value.ndim == 4:
                    value = value[:max(1, int(max_batch))]
                    limit = max(1, int(max_sequence_length))
                    if name in {"QK_LOGITS", "ATTENTION_PROBS"}:
                        value = value[:, :, :limit, :limit]
                    else:
                        value = value[:, :, :limit, :]
                elif value.ndim == 3:
                    value = value[:max(1, int(max_batch)), :max(1, int(max_sequence_length)), :]
                elif value.ndim >= 1:
                    value = value[:max(1, int(max_batch))]
                if cpu:
                    value = value.cpu()
                tensors[f"layers/{layer_index}/activations/{name}"] = value.contiguous().clone()
        return tensors

    def close(self) -> None:
        self.set_enabled(False)


def collect_csd_diagnostics(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    use_nesterov_score: bool = True,
) -> Dict[str, float]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    entries = []
    for group in optimizer.param_groups:
        if not bool(group.get("use_muon", False)):
            continue
        beta = float(group.get("momentum", 0.9))
        for parameter in group["params"]:
            if parameter.ndim < 2 or parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            previous = optimizer.state.get(parameter, {}).get("momentum_buffer")
            previous = previous.detach().float() if torch.is_tensor(previous) else torch.zeros_like(gradient)
            if use_nesterov_score:
                direction = previous.mul(beta * beta).add(gradient, alpha=1.0 - beta * beta)
            else:
                direction = previous.mul(beta).add(gradient, alpha=1.0 - beta)
            nuclear = torch.linalg.matrix_norm(direction.flatten(1), ord="nuc")
            entries.append((names.get(id(parameter), f"parameter_{len(entries)}"), nuclear))
    if not entries:
        return {}
    values_device = torch.stack([entry[1] for entry in entries])
    values = values_device.detach().cpu().double()
    denom_l2 = float(torch.linalg.vector_norm(values).item())
    denom_l1 = float(values.sum().item())
    result: Dict[str, float] = {
        "csd/objective": denom_l2,
        "csd/sum_nuclear_norms": denom_l1,
        "csd/matrix_count": float(len(entries)),
    }
    if denom_l1 > 0:
        shares = values / denom_l1
        result["csd/effective_matrix_count_l1"] = float(1.0 / shares.square().sum().item())
        result["csd/max_l1_share"] = float(shares.max().item())
        result["csd/share_entropy"] = _safe_float(-(shares * shares.clamp_min(1e-30).log()).sum())
    for (name, _nuclear_tensor), nuclear in zip(entries, values.tolist()):
        safe = name.replace(".", "/")
        result[f"csd/matrices/{safe}/nuclear_norm"] = float(nuclear)
        result[f"csd/matrices/{safe}/coefficient_l2"] = float(nuclear) / denom_l2 if denom_l2 > 0 else 0.0
        result[f"csd/matrices/{safe}/share_l1"] = float(nuclear) / denom_l1 if denom_l1 > 0 else 0.0
    return result


def optimizer_diagnostic_metrics(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for parameter_name, values in getattr(optimizer, "last_step_diagnostics", {}).items():
        safe_name = parameter_name.replace(".", "/")
        for metric_name, value in values.items():
            if value is not None and math.isfinite(float(value)):
                result[f"muon/matrices/{safe_name}/{metric_name}"] = float(value)
    return result


def build_tensor_snapshot(
    model: nn.Module,
    activation_recorder: GPT2ActivationRecorder,
    snapshot_layers: Sequence[int],
    extra_tensors: Optional[Mapping[str, torch.Tensor]] = None,
    include_weights: bool = True,
) -> Dict[str, torch.Tensor]:
    snapshot = (
        collect_weight_tensors(model, layer_indices=snapshot_layers, clone=True)
        if include_weights else {}
    )
    snapshot.update(activation_recorder.get_tensors(
        layer_indices=snapshot_layers, cpu=True, max_batch=1, max_sequence_length=128
    ))
    if extra_tensors:
        for name, tensor in extra_tensors.items():
            snapshot[name] = tensor.detach().cpu().contiguous().clone()
    return snapshot


def _tensor_image_view(tensor: torch.Tensor, name: str) -> Optional[torch.Tensor]:
    """Convert a captured tensor into a meaningful 2D view for heatmap logging.

    Weight/composition tensors are already 2D. Activations are typically
    [batch, heads, sequence, head_dim] or [batch, sequence, hidden_dim]. We
    display the first sample and stack heads vertically, preserving all heads
    instead of silently selecting only head zero.
    """
    value = tensor.detach().float().cpu()
    if value.ndim == 2:
        return value
    if value.ndim == 4:
        # [B, H, S, D] -> [H*S, D]. This also handles [B, H, S, S]
        # attention-logit/probability tensors.
        value = value[0]
        heads, rows, columns = value.shape
        return value.reshape(heads * rows, columns)
    if value.ndim == 3:
        # [B, S, D] -> [S, D].
        return value[0]
    return None


def _wandb_tensor_heatmap(wandb, tensor: torch.Tensor, name: str):
    """Render signed tensors as zero-centred diverging heatmaps and
    probabilities as sequential heatmaps. The source tensor is never changed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    view = _tensor_image_view(tensor, name)
    if view is None or view.numel() == 0:
        return None

    # Sanitize only the visualization copy; exact tensors remain in the .pt
    # artifact. This prevents NaN/Inf values from breaking imshow/norm.
    array = view.numpy()
    finite = torch.isfinite(view)
    if not bool(finite.any()):
        return None
    finite_values = view[finite]
    is_probability = any(
        token in name.lower()
        for token in ("attention_probs", "attention_probabilities")
    )

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    try:
        if is_probability:
            # Attention probabilities are nonnegative. A robust upper bound
            # reveals structure without letting a few peaks wash out the map.
            upper = float(torch.quantile(finite_values, 0.995).item())
            upper = max(upper, 1e-8)
            safe_array = array.copy()
            safe_array[~finite.numpy()] = 0.0
            im = ax.imshow(
                safe_array,
                cmap="viridis",
                norm=Normalize(vmin=0.0, vmax=upper, clip=True),
                aspect="auto",
                interpolation="nearest",
            )
        else:
            # Weights, activations and logits are signed: centre the colour
            # scale at zero so positive and negative values are comparable.
            scale = float(torch.quantile(finite_values.abs(), 0.995).item())
            scale = max(scale, 1e-8)
            safe_array = array.copy()
            safe_array[~finite.numpy()] = 0.0
            im = ax.imshow(
                safe_array,
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale),
                aspect="auto",
                interpolation="nearest",
            )

        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Feature / key position")
        ax.set_ylabel("Feature / query position (heads stacked for activations)")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        return wandb.Image(fig)
    finally:
        plt.close(fig)


def log_tensor_visuals(run: Any, snapshot: Mapping[str, torch.Tensor], step: int) -> None:
    """Log colour heatmaps and value histograms for saved weights/activations.

    Matrix tensors are rendered as scientific heatmaps (not W&B's default
    grayscale interpretation of a raw 2D NumPy array). Rank-3/4 activations
    are reduced to a 2D view using the first batch item and all heads stacked
    vertically. Histograms use finite raw values and deterministic subsampling.
    """
    import wandb

    payload: Dict[str, Any] = {}
    max_points = 50_000
    for name, tensor in snapshot.items():
        if "/weights/" not in name and "/activations/" not in name:
            continue

        value = tensor.detach().float().reshape(-1)
        if value.numel() == 0:
            continue
        value = value[torch.isfinite(value)]
        if value.numel() == 0:
            continue
        value = value.cpu()
        stride = max(1, math.ceil(value.numel() / max_points))
        payload[f"tensor_histograms/{name}"] = wandb.Histogram(
            value[::stride].numpy(), num_bins=64
        )

        heatmap = _wandb_tensor_heatmap(wandb, tensor, name)
        if heatmap is not None:
            payload[f"tensor_images/{name}"] = heatmap

    if payload:
        payload["diagnostics/tensor_snapshot_step"] = int(step)
        run.log(payload, step=step)


def log_tensor_artifact(
    run: Any,
    snapshot: Mapping[str, torch.Tensor],
    save_dir: str,
    step: int,
    epoch: int,
) -> str:
    import wandb

    artifact_dir = os.path.join(save_dir, "diagnostics")
    os.makedirs(artifact_dir, exist_ok=True)
    filename = f"lm_tensors_epoch_{int(epoch):04d}_step_{int(step):08d}.pt"
    path = os.path.join(artifact_dir, filename)
    torch.save({name: tensor.detach().cpu() for name, tensor in snapshot.items()}, path)
    artifact = wandb.Artifact(
        name=f"lm-matrix-tensors-{run.id}-e{int(epoch)}-s{int(step)}",
        type="matrix-diagnostics",
        metadata={
            "epoch": int(epoch),
            "global_step": int(step),
            "tensor_count": len(snapshot),
            "snapshot_layers": sorted({
                int(name.split("/")[1]) for name in snapshot
                if name.startswith("layers/") and name.split("/")[1].isdigit()
            }),
        },
    )
    artifact.add_file(path, name=filename)
    run.log_artifact(artifact)
    return path


def compare_weight_snapshots(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    """Quantify per-matrix geometry changes across a committed teleportation."""
    metrics: Dict[str, float] = {}
    for name in sorted(set(before) & set(after)):
        left = before[name].detach().float()
        right = after[name].detach().float()
        prefix = name.replace("/weights/", "/")
        left_norm = torch.linalg.vector_norm(left)
        right_norm = torch.linalg.vector_norm(right)
        delta_norm = torch.linalg.vector_norm(right - left)
        metrics[f"teleport_geometry/{prefix}/relative_fro_change"] = _safe_float(
            delta_norm / left_norm.clamp_min(1e-12)
        )
        metrics[f"teleport_geometry/{prefix}/fro_ratio_after_before"] = _safe_float(
            right_norm / left_norm.clamp_min(1e-12)
        )
        metrics[f"teleport_geometry/{prefix}/cosine_before_after"] = cosine_similarity(left, right)
    return metrics


def collect_global_gradient_metrics(model: nn.Module) -> Dict[str, float]:
    squared_terms = []
    parameter_norms = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        norm = torch.linalg.vector_norm(parameter.grad.detach().float())
        squared_terms.append(norm.square())
        parameter_norms.append(norm)
    if not parameter_norms:
        return {
            "gradients/global_frobenius": 0.0,
            "gradients/max_parameter_frobenius": 0.0,
            "gradients/active_parameter_count": 0.0,
        }
    summary = torch.stack((torch.stack(squared_terms).sum().sqrt(), torch.stack(parameter_norms).max())).detach().cpu().tolist()
    return {
        "gradients/global_frobenius": float(summary[0]),
        "gradients/max_parameter_frobenius": float(summary[1]),
        "gradients/active_parameter_count": float(len(parameter_norms)),
    }
