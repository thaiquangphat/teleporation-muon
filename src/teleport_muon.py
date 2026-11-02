"""CSD-based Muon teleportation implementation.

This module is deliberately separate from ``teleport.py``. It implements the
symmetry-search path selected by the existing ``--tele-method muon_alignment``
CLI option, while keeping that public interface unchanged.

At scheduled teleportation minibatches, candidates are ranked by a constrained
steepest-descent (CSD) linearized-decrease score. An accepted non-identity
symmetry is committed and the following optimizer step is the repository's
ordinary Muon step at its unmodified learning rate. No Momo scalar model,
model truncation, or adaptive learning-rate multiplier is used.

All non-scheduled minibatches use the unmodified Muon/Adam optimizer. The CSD
score uses the same Nesterov momentum direction that the repository's Muon
implementation passes to its Newton--Schulz polar-factor approximation.
"""

import copy
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from muon import create_muon_optimizer

def generate_tele_scheduler(tele_batch, number_of_batch, tele_opt, tele_cons):
    if tele_opt == 0:
        tele_scheduler = [True] * tele_batch + [False] * (number_of_batch - tele_batch)
        random.shuffle(tele_scheduler)
    elif tele_opt == 1:
        tele_scheduler = [False] * number_of_batch
        count = 0
        for i in range(0, (tele_batch + tele_cons - 1) // tele_cons):
            for j in range(tele_cons):
                if ((i * number_of_batch) // ((tele_batch + tele_cons - 1) // tele_cons)) + j < number_of_batch and count < tele_batch:
                    tele_scheduler[((i * number_of_batch) // ((tele_batch + tele_cons - 1) // tele_cons)) + j] = True
                    count += 1
    return tele_scheduler


def gen_inv_matrix(n_head, d_head, tele_high, tele_low, tele_sign):
    size = d_head
    dtype = torch.float32
    M = torch.eye(size, dtype=dtype)
    M *= torch.rand(size, dtype=dtype) * (tele_high - tele_low) + tele_low
    if tele_sign == 1:
        mask = torch.randint(0, 2, (size,), dtype=dtype) * 2 - 1
        M[range(size), range(size)] *= mask
    indices = torch.randperm(size)
    M = M[indices, :]
    indices = torch.randperm(size)
    M = M[:, indices]
    M_inv = torch.transpose(M, -1, -2)
    M_inv = torch.where(M_inv != 0, 1.0 / M_inv, M_inv)
    M = torch.block_diag(*[M] * n_head)
    M_inv = torch.block_diag(*[M_inv] * n_head)
    return M, M_inv


def gen_rope_inv_matrix(n_head, d_head, tele_high, tele_low, tele_sign):
    if d_head % 2 != 0:
        raise ValueError(f"d_head must be even for RoPE structure, got d_head={d_head}")
    dtype = torch.float32
    two_pi = 2.0 * torch.pi

    def make_head_blocks():
        blocks = []
        inv_blocks = []
        thetas = torch.rand(d_head // 2, dtype=dtype) * two_pi
        rhos = torch.rand(d_head // 2, dtype=dtype) * (tele_high - tele_low) + tele_low
        if tele_sign == 1:
            signs = (torch.randint(0, 2, (d_head // 2,)) * 2 - 1).to(dtype)
            rhos = rhos * signs
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)
        for c, s, r in zip(cos, sin, rhos):
            M_block = torch.tensor([[r * c, -r * s], [r * s, r * c]], dtype=dtype)
            rinv = 1.0 / r
            M_inv_block = torch.tensor([[rinv * c, rinv * s], [-rinv * s, rinv * c]], dtype=dtype)
            blocks.append(M_block)
            inv_blocks.append(M_inv_block)
        return torch.block_diag(*blocks), torch.block_diag(*inv_blocks)

    per_head_M = []
    per_head_Minv = []
    for _ in range(n_head):
        Mh, Minvh = make_head_blocks()
        per_head_M.append(Mh)
        per_head_Minv.append(Minvh)
    M = torch.block_diag(*per_head_M)
    M_inv = torch.block_diag(*per_head_Minv)
    return M, M_inv


def teleportation_att(vit_model, M, M_inv, tele_layer):
    """Apply the MNISTCIFAR Q/K symmetry while preserving this public API.

    For row-vector activations, this implements Q' = M Q and
    K' = M^{-T} K, with the corresponding bias transformations.
    """
    layers = vit_model.vit.encoder.layer
    if tele_layer == "all":
        selected_layers = layers
    elif tele_layer == "first":
        selected_layers = [layers[0]]
    elif tele_layer == "last":
        selected_layers = [layers[-1]]
    else:
        raise ValueError(f"Unsupported tele_layer: {tele_layer}")

    for block in selected_layers:
        q_linear = block.attention.attention.query
        k_linear = block.attention.attention.key
        q_transform = M.to(device=q_linear.weight.device, dtype=q_linear.weight.dtype)
        k_inverse = M_inv.to(device=k_linear.weight.device, dtype=k_linear.weight.dtype)
        with torch.no_grad():
            q_linear.weight.copy_(q_transform @ q_linear.weight)
            if q_linear.bias is not None:
                q_linear.bias.copy_(q_linear.bias @ q_transform.mT)
            k_linear.weight.copy_(k_inverse.mT @ k_linear.weight)
            if k_linear.bias is not None:
                k_linear.bias.copy_(k_linear.bias @ k_inverse)


def teleportation_mlp(vit_model, M, M_inv):
    """Apply the MNISTCIFAR hidden-unit symmetry to every MLP block.

    For ReLU networks, M must be a positive monomial matrix (a permutation
    combined with positive diagonal scaling); candidate generation below
    enforces that by disabling signed MLP scales.
    """
    layers = vit_model.vit.encoder.layer
    for block in layers:
        intermediate_linear = block.intermediate.dense
        output_linear = block.output.dense
        transform = M.to(device=intermediate_linear.weight.device, dtype=intermediate_linear.weight.dtype)
        inverse = M_inv.to(device=output_linear.weight.device, dtype=output_linear.weight.dtype)
        with torch.no_grad():
            intermediate_linear.weight.copy_(transform @ intermediate_linear.weight)
            if intermediate_linear.bias is not None:
                intermediate_linear.bias.copy_(intermediate_linear.bias @ transform.mT)
            output_linear.weight.copy_(output_linear.weight @ inverse)


def validate_bounds(low: float, high: float) -> Tuple[float, float]:
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ValueError(f"Teleport bounds must satisfy 0 < low < high; got low={low}, high={high}.")
    return low, high


def attention_blocks(model: nn.Module, tele_layer: str) -> List[nn.Module]:
    layers = list(model.vit.encoder.layer)
    if tele_layer == "all":
        return layers
    if tele_layer == "first":
        return [layers[0]]
    if tele_layer == "last":
        return [layers[-1]]
    raise ValueError(f"Unsupported tele_layer: {tele_layer}")


def qk_linear_modules(model: nn.Module, tele_layer: str) -> List[Tuple[nn.Linear, nn.Linear]]:
    pairs = []
    for block in attention_blocks(model, tele_layer):
        pairs.append((block.attention.attention.query, block.attention.attention.key))
    return pairs


def qk_parameter_items(model: nn.Module, tele_layer: str) -> List[Tuple[str, nn.Parameter]]:
    selected = {id(module) for pair in qk_linear_modules(model, tele_layer) for module in pair}
    items = []
    for name, module in model.named_modules():
        if id(module) in selected:
            items.append((f"{name}.weight", module.weight))
            if module.bias is not None:
                items.append((f"{name}.bias", module.bias))
    return items


def _matrix_scale_summary(matrix: Optional[torch.Tensor]) -> List[float]:
    if matrix is None:
        return []
    # The row norms are the actual per-coordinate scale factors for the
    # monomial matrices and rho*rotation blocks used by the source framework.
    return torch.linalg.vector_norm(matrix.float(), ord=2, dim=1).detach().cpu().tolist()


def generate_symmetry_candidates(
    model: nn.Module,
    args: Any,
    low: float,
    high: float,
    count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate identity plus MNISTCIFAR symmetry-preserving transforms.

    Keeps count semantics (identity included) and uses the existing CLI flags.
    The sampled attention transform is a general monomial matrix without RoPE,
    or a block-diagonal scaled rotation when RoPE is active. MLP candidates use
    positive monomial matrices so the ReLU hidden-unit symmetry is preserved.
    """
    validate_bounds(low, high)
    use_attention = bool(getattr(args, "tele_att", 1))
    use_mlp = bool(getattr(args, "tele_mlp", 0))
    if not use_attention and not use_mlp:
        raise ValueError("muon_alignment requires --tele-att 1, --tele-mlp 1, or both.")

    layers = list(model.vit.encoder.layer)
    if use_attention:
        q_linear = qk_linear_modules(model, args.tele_layer)[0][0]
        d_model = q_linear.out_features
        n_head = int(args.num_heads)
        if n_head <= 0 or d_model % n_head != 0:
            raise ValueError(f"Q/K output size {d_model} must be divisible by num_heads={n_head}.")
        d_head = d_model // n_head
        rope = getattr(args, "position_embedding", "learnable") == "rope"
    if use_mlp:
        mlp_dim = layers[0].intermediate.dense.out_features

    original_rng = capture_rng_state()
    candidates: List[Dict[str, Any]] = []
    try:
        # Candidate generation uses the same source-file matrix generators but
        # is deterministic per (seed, step) and leaves the caller RNG untouched.
        torch.manual_seed(int(seed))
        for index in range(max(0, int(count))):
            candidate: Dict[str, Any] = {
                "identity": index == 0,
                "M_att": None,
                "M_att_inv": None,
                "M_mlp": None,
                "M_mlp_inv": None,
            }
            if use_attention:
                if index == 0:
                    matrix = torch.eye(d_model, dtype=torch.float32)
                    matrix_inv = matrix.clone()
                elif rope:
                    matrix, matrix_inv = gen_rope_inv_matrix(
                        n_head=n_head,
                        d_head=d_head,
                        tele_high=high,
                        tele_low=low,
                        tele_sign=int(getattr(args, "tele_sign", 0)),
                    )
                else:
                    matrix, matrix_inv = gen_inv_matrix(
                        n_head=n_head,
                        d_head=d_head,
                        tele_high=high,
                        tele_low=low,
                        tele_sign=int(getattr(args, "tele_sign", 0)),
                    )
                candidate["M_att"] = matrix
                candidate["M_att_inv"] = matrix_inv
            if use_mlp:
                if index == 0:
                    matrix = torch.eye(mlp_dim, dtype=torch.float32)
                    matrix_inv = matrix.clone()
                else:
                    # ReLU is not sign-equivariant. Positive monomial matrices
                    # are the valid hidden-unit symmetry, so MLP sign flips are
                    # intentionally excluded while preserving the same API.
                    matrix, matrix_inv = gen_inv_matrix(
                        n_head=1,
                        d_head=mlp_dim,
                        tele_high=high,
                        tele_low=low,
                        tele_sign=0,
                    )
                candidate["M_mlp"] = matrix
                candidate["M_mlp_inv"] = matrix_inv
            candidate["attention_scale_summary"] = _matrix_scale_summary(candidate["M_att"])
            candidate["mlp_scale_summary"] = _matrix_scale_summary(candidate["M_mlp"])
            # Preserve the legacy `scale` field as the primary transform's
            # coordinate scales, while aggregate bounds cover every active family.
            candidate["scale_summary"] = (
                candidate["attention_scale_summary"]
                if use_attention
                else candidate["mlp_scale_summary"]
            )
            all_scales = candidate["attention_scale_summary"] + candidate["mlp_scale_summary"]
            candidate["scale_min"] = min(all_scales) if all_scales else 1.0
            candidate["scale_max"] = max(all_scales) if all_scales else 1.0
            candidates.append(candidate)
    finally:
        restore_rng_state(original_rng)
    if not candidates or not candidates[0]["identity"]:
        raise ValueError("At least one candidate (the identity) is required.")
    return candidates


def apply_symmetry_candidate(
    model: nn.Module,
    candidate: Dict[str, Any],
    args: Any,
) -> None:
    """Apply a generated candidate using the unchanged MNISTCIFAR APIs."""
    if bool(getattr(args, "tele_att", 1)):
        teleportation_att(model, candidate["M_att"], candidate["M_att_inv"], args.tele_layer)
    if bool(getattr(args, "tele_mlp", 0)):
        teleportation_mlp(model, candidate["M_mlp"], candidate["M_mlp_inv"])


def teleportation_parameter_items(model: nn.Module, args: Any) -> List[Tuple[str, nn.Parameter]]:
    """Parameters whose coordinate systems change under the selected transform."""
    items: List[Tuple[str, nn.Parameter]] = []
    if bool(getattr(args, "tele_att", 1)):
        items.extend(qk_parameter_items(model, args.tele_layer))
    if bool(getattr(args, "tele_mlp", 0)):
        for i, block in enumerate(model.vit.encoder.layer):
            prefix = f"vit.encoder.layer.{i}"
            intermediate = block.intermediate.dense
            output = block.output.dense
            items.append((f"{prefix}.intermediate.dense.weight", intermediate.weight))
            if intermediate.bias is not None:
                items.append((f"{prefix}.intermediate.dense.bias", intermediate.bias))
            items.append((f"{prefix}.output.dense.weight", output.weight))
    # De-duplicate in case a future model shares a parameter in two modules.
    unique: List[Tuple[str, nn.Parameter]] = []
    seen = set()
    for name, parameter in items:
        if id(parameter) not in seen:
            unique.append((name, parameter))
            seen.add(id(parameter))
    return unique


def _transform_momentum_left(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    transform: torch.Tensor,
) -> None:
    """Transport a Muon momentum buffer under a left-multiplied weight transform."""
    state = optimizer.state.get(parameter)
    if not state:
        return
    momentum = state.get("momentum_buffer")
    if not torch.is_tensor(momentum):
        return
    matrix = transform.to(device=momentum.device, dtype=momentum.dtype)
    with torch.no_grad():
        momentum.copy_(matrix @ momentum)


def _transform_momentum_right(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    transform: torch.Tensor,
) -> None:
    """Transport a Muon momentum buffer under a right-multiplied weight transform."""
    state = optimizer.state.get(parameter)
    if not state:
        return
    momentum = state.get("momentum_buffer")
    if not torch.is_tensor(momentum):
        return
    matrix = transform.to(device=momentum.device, dtype=momentum.dtype)
    with torch.no_grad():
        momentum.copy_(momentum @ matrix)


def transport_teleportation_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    candidate: Dict[str, Any],
    args: Any,
) -> None:
    """Transport Muon momentum through a symmetry, resetting only transformed Adam states.

    For Q/K, W_Q' = M W_Q and W_K' = M^{-T} W_K, so their gradient-like
    momentum buffers transform as M^{-T} B_Q and M B_K. For the MLP symmetry,
    W_1' = M W_1 and W_2' = W_2 M^{-1}, giving M^{-T} B_1 and B_2 M^T.

    Adam's elementwise second-moment estimates cannot in general be transported
    exactly through these coordinate changes. The states for transformed biases
    (which use the auxiliary Adam group) are therefore cleared, while the Muon
    weight momentum buffers are preserved in the new coordinates.
    """
    if bool(candidate.get("identity", False)):
        return

    if bool(getattr(args, "tele_att", 1)):
        matrix = candidate["M_att"]
        matrix_inv = candidate["M_att_inv"]
        for query, key in qk_linear_modules(model, args.tele_layer):
            _transform_momentum_left(optimizer, query.weight, matrix_inv.mT)
            _transform_momentum_left(optimizer, key.weight, matrix)
            if query.bias is not None:
                optimizer.state.pop(query.bias, None)
            if key.bias is not None:
                optimizer.state.pop(key.bias, None)

    if bool(getattr(args, "tele_mlp", 0)):
        matrix = candidate["M_mlp"]
        matrix_inv = candidate["M_mlp_inv"]
        for block in model.vit.encoder.layer:
            intermediate = block.intermediate.dense
            output = block.output.dense
            _transform_momentum_left(optimizer, intermediate.weight, matrix_inv.mT)
            _transform_momentum_right(optimizer, output.weight, matrix.mT)
            if intermediate.bias is not None:
                optimizer.state.pop(intermediate.bias, None)


def _muon_groups(optimizer: torch.optim.Optimizer) -> List[dict]:
    """Return Muon parameter groups and validate their momentum convention."""
    groups = [group for group in optimizer.param_groups if bool(group.get("use_muon", False))]
    if not groups:
        raise ValueError(
            "CSD-Muon teleportation requires at least one optimizer group with use_muon=True."
        )

    betas = [float(group.get("momentum", 0.95)) for group in groups]
    beta = betas[0]
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"Muon momentum must lie in [0, 1); got {beta}.")
    if any(abs(value - beta) > 1e-12 for value in betas[1:]):
        raise ValueError(
            "CSD candidate scoring requires a common momentum coefficient "
            "across Muon parameter groups."
        )
    for group in groups:
        lr = float(group.get("lr", 0.0))
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError(f"Muon learning rates must be finite and nonnegative; got {lr}.")
    return groups


def _muon_matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    """Use the matrix view used by Muon for nuclear-norm calculations."""
    value = tensor.detach().float()
    if value.ndim == 2:
        return value
    if value.ndim >= 3:
        # The repository's Muon implementation flattens convolutional
        # tensors along dimension 1 before applying Newton--Schulz.
        return value.flatten(1)
    raise ValueError(
        "A parameter in a Muon group is not matrix-valued; "
        f"received a tensor with shape {tuple(tensor.shape)}."
    )


def _csd_metrics_from_loss(
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    args: Any,
    step_index: Optional[int] = None,
) -> Dict[str, float]:
    """Compute the candidate's CSD linearized-decrease score.

    For each Muon matrix parameter, predict the repository momentum buffer

        B_t = beta B_{t-1} + (1-beta) G_t,

    and its actual Nesterov direction

        V_t = (1-beta) G_t + beta B_t.

    The spectral-norm CSD LMO is the negative polar factor. Its ideal
    first-order decrease for a layer is eta * ||V_t||_*. Summing over the
    Muon matrix groups gives the candidate score used for symmetry selection.
    No loss lower bound or adaptive LR scaling is involved.
    """
    groups = _muon_groups(optimizer)
    beta = float(groups[0].get("momentum", 0.95))
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"Muon momentum must lie in [0, 1); got {beta}.")

    loss_value = float(loss.detach().float().item())
    momentum_dual_norm = 0.0
    nesterov_dual_norm = 0.0
    lr_weighted_dual_norm = 0.0
    active_parameters = 0

    for group in groups:
        group_lr = float(group.get("lr", 0.0))
        group_beta = float(group.get("momentum", beta))
        if abs(group_beta - beta) > 1e-12:
            raise ValueError("CSD candidate scoring requires a common Muon momentum coefficient.")

        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None or parameter.ndim < 2:
                continue

            state = optimizer.state.get(parameter, {})
            previous_momentum = state.get("momentum_buffer")
            grad32 = gradient.detach().float()
            if torch.is_tensor(previous_momentum):
                previous32 = previous_momentum.detach().float()
                momentum = previous32.mul(group_beta).add(grad32, alpha=1.0 - group_beta)
            else:
                momentum = grad32.mul(1.0 - group_beta)

            # Matches muon.py: update = gradient.lerp(momentum_buffer, beta).
            nesterov_direction = torch.lerp(grad32, momentum, group_beta)

            momentum_norm = float(
                torch.linalg.matrix_norm(_muon_matrix_view(momentum), ord="nuc").item()
            )
            nesterov_norm = float(
                torch.linalg.matrix_norm(_muon_matrix_view(nesterov_direction), ord="nuc").item()
            )
            momentum_dual_norm += momentum_norm
            nesterov_dual_norm += nesterov_norm
            lr_weighted_dual_norm += group_lr * nesterov_norm
            active_parameters += 1

    if active_parameters == 0:
        raise RuntimeError("No Muon matrix parameters received gradients; cannot score CSD candidates.")
    if not math.isfinite(lr_weighted_dual_norm):
        raise ValueError("Non-finite CSD candidate score.")

    return {
        "score": float(lr_weighted_dual_norm),
        "loss": loss_value,
        "momentum_dual_norm": float(momentum_dual_norm),
        "dual_norm": float(nesterov_dual_norm),
        "nesterov_dual_norm": float(nesterov_dual_norm),
        "lr_weighted_dual_norm": float(lr_weighted_dual_norm),
        "lr_scale": 1.0,
        "beta": float(beta),
        "active_parameters": float(active_parameters),
        "step_index": float(step_index) if step_index is not None else -1.0,
    }


def compute_symmetry_muon_alignment(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
) -> float:
    """Return the CSD predicted-decrease score for one candidate.

    The public signature and scalar return type are retained for compatibility.
    """
    optimizer.zero_grad(set_to_none=True)
    logits = model(pixel_values=batch[0]).logits
    loss = criterion(logits, batch[1])
    loss.backward()
    return _csd_metrics_from_loss(optimizer, loss, args)["score"]


def update_momo_for_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    step_index: int,
    args: Any,
) -> Dict[str, float]:
    """Backward-compatible no-op; the CSD variant has no Momo state.

    Kept with its original signature so older launchers importing this symbol
    do not fail. The CSD launcher does not call it.
    """
    return {
        "score": 0.0,
        "lr_scale": 1.0,
        "already_updated": 1.0,
        "csd_mode": 1.0,
    }


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def snapshot_live_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    optimizer_state = {}
    for name, parameter in named_params.items():
        raw_state = optimizer.state.get(parameter)
        if not raw_state:
            continue
        cloned = {}
        for key, value in raw_state.items():
            cloned[key] = value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        optimizer_state[name] = cloned
    return {
        "training": model.training,
        "params": {name: parameter.detach().clone() for name, parameter in named_params.items()},
        "grads": {
            name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in named_params.items()
        },
        "buffers": {name: buffer.detach().clone() for name, buffer in named_buffers.items()},
        "optimizer_state": optimizer_state,
        "optimizer_groups": [
            {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
    }


def assert_live_state_unchanged(model: nn.Module, optimizer: torch.optim.Optimizer, snapshot: Dict[str, Any]) -> None:
    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    if model.training != snapshot["training"]:
        raise RuntimeError("Virtual teleportation mutated live training/eval mode.")
    for name, parameter in named_params.items():
        if not torch.equal(parameter.detach(), snapshot["params"][name]):
            raise RuntimeError(f"Virtual teleportation mutated live parameter {name}.")
        live_grad = parameter.grad
        snap_grad = snapshot["grads"][name]
        if live_grad is None and snap_grad is None:
            continue
        if live_grad is None or snap_grad is None or not torch.equal(live_grad, snap_grad):
            raise RuntimeError(f"Virtual teleportation mutated live gradient {name}.")
    for name, buffer in named_buffers.items():
        if not torch.equal(buffer, snapshot["buffers"][name]):
            raise RuntimeError(f"Virtual teleportation mutated live buffer {name}.")
    for name, parameter in named_params.items():
        live_state = optimizer.state.get(parameter, {})
        snap_state = snapshot["optimizer_state"].get(name, {})
        if set(live_state) != set(snap_state):
            raise RuntimeError(f"Virtual teleportation mutated optimizer state keys for {name}.")
        for key, snap_value in snap_state.items():
            live_value = live_state[key]
            if torch.is_tensor(snap_value):
                if not torch.equal(live_value, snap_value):
                    raise RuntimeError(f"Virtual teleportation mutated optimizer state {name}.{key}.")
            elif live_value != snap_value:
                raise RuntimeError(f"Virtual teleportation mutated optimizer state {name}.{key}.")


def _make_optimizer_for_model(model: nn.Module, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    if isinstance(optimizer, torch.optim.AdamW):
        group = optimizer.param_groups[0]
        return torch.optim.AdamW(
            model.parameters(),
            lr=group["lr"],
            betas=group.get("betas", (0.9, 0.999)),
            eps=group.get("eps", 1e-8),
            weight_decay=group.get("weight_decay", 0.0),
        )
    if isinstance(optimizer, torch.optim.SGD):
        group = optimizer.param_groups[0]
        return torch.optim.SGD(
            model.parameters(),
            lr=group["lr"],
            momentum=group.get("momentum", 0.0),
            weight_decay=group.get("weight_decay", 0.0),
        )
    first = optimizer.param_groups[0]
    clone = create_muon_optimizer(
        model,
        lr=first["lr"],
        weight_decay=first.get("weight_decay", 0.0),
        momentum=first.get("momentum", 0.95),
    )
    clone_state_groups_from(optimizer, clone)
    return clone


def clone_state_groups_from(source: torch.optim.Optimizer, dest: torch.optim.Optimizer) -> None:
    if len(source.param_groups) != len(dest.param_groups):
        return
    for source_group, dest_group in zip(source.param_groups, dest.param_groups):
        for key, value in source_group.items():
            if key != "params":
                dest_group[key] = copy.deepcopy(value)


def clone_optimizer_state(model: nn.Module, model_clone: nn.Module, optimizer: torch.optim.Optimizer, optimizer_clone: torch.optim.Optimizer) -> None:
    real_named = dict(model.named_parameters())
    clone_named = dict(model_clone.named_parameters())
    for name, parameter in real_named.items():
        clone_parameter = clone_named[name]
        state = optimizer.state.get(parameter)
        if not state:
            continue
        cloned_state = {}
        for key, value in state.items():
            cloned_state[key] = value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        optimizer_clone.state[clone_parameter] = cloned_state


def clone_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[nn.Module, torch.optim.Optimizer]:
    model_clone = copy.deepcopy(model)
    optimizer_clone = _make_optimizer_for_model(model_clone, optimizer)
    clone_optimizer_state(model, model_clone, optimizer, optimizer_clone)
    return model_clone, optimizer_clone


def try_muon_alignment_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Select a symmetry by CSD predicted decrease and transport its state.

    Each candidate is evaluated on an isolated model/optimizer clone. Its
    score is the sum over Muon matrix parameters of

        eta_l * ||V_t^(l)||_*,

    where V_t is the same Nesterov direction used by the repository Muon
    implementation. This is the ideal first-order decrease of the associated
    spectral-norm CSD step (with the polar factor as the LMO).

    A non-identity candidate is committed only if it beats identity by
    ``tele_accept_tol``. The normal training loop then performs its ordinary
    Muon optimizer step with the unchanged base learning rate. This function
    never changes optimizer learning rates or maintains a Momo scalar model.
    """
    started = time.perf_counter()
    rng_state = capture_rng_state()
    live_snapshot = snapshot_live_state(model, optimizer)
    low, high = validate_bounds(args.tele_low, args.tele_high)
    if args.tele_candidates < 2:
        raise ValueError(
            "muon_alignment requires at least one non-identity candidate; "
            "--tele-candidates must be >= 2."
        )

    _muon_groups(optimizer)
    was_training = model.training
    committed = False
    model.train()
    try:
        candidates = generate_symmetry_candidates(
            model=model,
            args=args,
            low=low,
            high=high,
            count=args.tele_candidates,
            seed=args.seed + step_index,
        )

        results: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            candidate_model, candidate_optimizer = clone_model_and_optimizer(model, optimizer)
            # Use exactly the same stochastic minibatch realization per candidate.
            restore_rng_state(rng_state)
            if not bool(candidate["identity"]):
                apply_symmetry_candidate(candidate_model, candidate, args)
                transport_teleportation_optimizer_state(
                    candidate_model, candidate_optimizer, candidate, args
                )

            candidate_optimizer.zero_grad(set_to_none=True)
            logits = candidate_model(pixel_values=batch[0]).logits
            loss = criterion(logits, batch[1])
            loss.backward()
            metrics = _csd_metrics_from_loss(
                candidate_optimizer, loss, args, step_index=step_index
            )
            if not math.isfinite(float(metrics["score"])):
                raise ValueError(
                    f"Non-finite CSD score for candidate {index}: {metrics['score']}."
                )

            results.append(
                {
                    "candidate_index": index,
                    "identity": bool(candidate["identity"]),
                    "scale": candidate["scale_summary"],
                    "scale_min": float(candidate["scale_min"]),
                    "scale_max": float(candidate["scale_max"]),
                    "attention_scale": candidate["attention_scale_summary"],
                    "mlp_scale": candidate["mlp_scale_summary"],
                    "transform_type": "mnistcifar_symmetry_preserving_matrix",
                    "uses_attention": bool(getattr(args, "tele_att", 1)),
                    "uses_mlp": bool(getattr(args, "tele_mlp", 0)),
                    # Keep the legacy alignment field for existing W&B tables.
                    "alignment": float(metrics["score"]),
                    "csd_predicted_decrease": float(metrics["score"]),
                    "csd_momentum_dual_norm": float(metrics["momentum_dual_norm"]),
                    "csd_nesterov_dual_norm": float(metrics["nesterov_dual_norm"]),
                    "csd_lr_weighted_dual_norm": float(metrics["lr_weighted_dual_norm"]),
                    "csd_beta": float(metrics["beta"]),
                    "csd_active_parameters": int(metrics["active_parameters"]),
                    "predicted_first_order_decrease": float(metrics["score"]),
                    "predicted_decrease_evaluated": True,
                    "alignment_evaluated": True,
                    "alignment_rank": None,
                    "virtual_loss_evaluated": False,
                    "post_step_loss": None,
                    "searched": False,
                }
            )

        ranked = sorted(
            results,
            key=lambda item: (-float(item["csd_predicted_decrease"]), int(item["candidate_index"])),
        )
        for rank, item in enumerate(ranked, start=1):
            item["alignment_rank"] = rank

        identity_result = next(item for item in results if item["identity"])
        non_identity_results = [item for item in results if not item["identity"]]
        best_non_identity = max(
            non_identity_results,
            key=lambda item: float(item["csd_predicted_decrease"]),
        ) if non_identity_results else None

        accept_tol = float(getattr(args, "tele_accept_tol", 0.0))
        if not math.isfinite(accept_tol) or accept_tol < 0.0:
            raise ValueError(
                f"tele_accept_tol must be finite and nonnegative; got {accept_tol}."
            )
        identity_score = float(identity_result["csd_predicted_decrease"])
        accepted = (
            best_non_identity is not None
            and float(best_non_identity["csd_predicted_decrease"]) > identity_score + accept_tol
        )
        selected = best_non_identity if accepted else identity_result
        selected_index = int(selected["candidate_index"])
        selected_candidate = candidates[selected_index]

        # Candidate search must not alter the live training state.
        restore_rng_state(rng_state)
        model.train(was_training)
        assert_live_state_unchanged(model, optimizer, live_snapshot)

        if accepted:
            apply_symmetry_candidate(model, selected_candidate, args)
            transport_teleportation_optimizer_state(
                model, optimizer, selected_candidate, args
            )
            committed = True
            reason = "selected_max_csd_predicted_decrease"
        else:
            reason = "identity_selected_max_csd_predicted_decrease_noop"

        selected_score = float(selected["csd_predicted_decrease"])
        score_gain = selected_score - identity_score
        search_time = round(time.perf_counter() - started, 6)

        telemetry = {
            "method": "csd_alignment",
            "accepted": accepted,
            "reason": reason,
            "step": step_index,
            "search_time_seconds": search_time,
            "regular_muon_post_step_loss": None,
            "identity_reset_loss": None,
            "identity_alignment": identity_score,
            "identity_paper_momentum_linearized_decrease": identity_score,
            "best_non_identity_candidate_loss": None,
            "beats_regular_baseline": None,
            "beats_reset_only": None,
            "accepted_beyond_reset_only": None,
            "selected_alignment": selected_score,
            "selected_paper_momentum_linearized_decrease": selected_score,
            "selected_post_step_loss": None,
            "selected_identity": not accepted,
            "selected_candidate_index": selected_index,
            "selected_scale": selected["scale"],
            "scale_min": selected["scale_min"],
            "scale_max": selected["scale_max"],
            "selected_attention_scale": selected_candidate["attention_scale_summary"],
            "selected_mlp_scale": selected_candidate["mlp_scale_summary"],
            "identity_rank": int(identity_result["alignment_rank"]),
            "state_reset": False,
            "muon_momentum_reset": False,
            "auxiliary_adam_state_reset": accepted,
            "momentum_state_transported": accepted,
            "selection_objective": "constrained_steepest_descent",
            "paper_objective_definition": (
                "sum_muon_groups(lr * sum_l nuclear_norm(nesterov_direction_l))"
            ),
            "csd_predicted_decrease": float(selected["csd_predicted_decrease"]),
            "csd_momentum_dual_norm": float(selected["csd_momentum_dual_norm"]),
            "csd_nesterov_dual_norm": float(selected["csd_nesterov_dual_norm"]),
            "csd_lr_weighted_dual_norm": float(selected["csd_lr_weighted_dual_norm"]),
            "csd_lr_multiplier_applied": 1.0,
            "adaptive_lr_scaling_applied": False,
            "tele_accept_tol": accept_tol,
            "csd_beta": float(selected["csd_beta"]),
            "identity_predicted_first_order_decrease": identity_score,
            "selected_predicted_first_order_decrease": selected_score,
            "tele_layer": args.tele_layer,
            "transform_family": "mnistcifar_symmetry_preserving_matrix",
            "uses_attention": bool(getattr(args, "tele_att", 1)),
            "uses_mlp": bool(getattr(args, "tele_mlp", 0)),
            "alignment_gain": score_gain,
            "post_loss_gain": None,
            "proposed_candidate_index": (
                int(best_non_identity["candidate_index"])
                if best_non_identity is not None else None
            ),
            "proposed_post_loss_gain": None,
            "accepted_candidate_index": selected_index if accepted else None,
            "accepted_post_loss_gain": None,
            "shortlist_size": None,
            "num_candidates_evaluated": len(results),
            "results": results,
        }
        return accepted, telemetry
    finally:
        restore_rng_state(rng_state)
        model.train(was_training)
        if not committed:
            assert_live_state_unchanged(model, optimizer, live_snapshot)


def try_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """CSD-based Muon teleportation dispatcher; rejects the naive path."""
    method = getattr(args, "tele_method", "muon_alignment")
    if method != "muon_alignment":
        raise ValueError(
            "train-muon.py supports only --tele-method muon_alignment; "
            "use the original train.py for naive or plain runs."
        )
    return try_muon_alignment_teleportation(
        model, optimizer, batch, criterion, args, step_index
    )


__all__ = [
    "generate_tele_scheduler",
    "try_teleportation",
    "try_muon_alignment_teleportation",
    "update_momo_for_step",
    "compute_symmetry_muon_alignment",
]

