"""CSD-Muon teleportation for the GPT-2 WikiText implementation.

Symmetries are applied to the *fused* GPT-2 QKV projection by transforming
only its Q and K output-coordinate blocks. Attention value coordinates and the
attention output projection are untouched. The optional GELU-MLP symmetry is
permutation-only (no scaling), since GELU is not positively homogeneous.
"""

from __future__ import annotations

import copy
import math
import random
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

try:  # package import
    from .muon import SingleDeviceMuonWithAuxAdam
except ImportError:  # ``python src/train_lm.py`` import path
    from muon import SingleDeviceMuonWithAuxAdam


def _muon_groups(optimizer: torch.optim.Optimizer) -> List[dict]:
    groups = [group for group in optimizer.param_groups if bool(group.get("use_muon", False))]
    if not groups:
        raise ValueError("CSD-Muon requires an optimizer group with use_muon=True")
    betas = [float(group.get("momentum", 0.9)) for group in groups]
    if any(abs(value - betas[0]) > 1e-12 for value in betas[1:]):
        raise ValueError("CSD scoring requires a common Muon momentum coefficient")
    if not (0.0 <= betas[0] < 1.0):
        raise ValueError("Muon momentum must lie in [0,1)")
    return groups


def _muon_matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    if value.ndim == 2:
        return value
    if value.ndim >= 3:
        return value.flatten(1)
    raise ValueError(f"Muon parameter must be matrix-valued, got {tuple(tensor.shape)}")


def _csd_metrics_from_loss(optimizer, loss, args=None, step_index=None) -> Dict[str, float]:
    """Same Nesterov-direction nuclear-norm score as src/teleport_muon.py."""
    groups = _muon_groups(optimizer)
    beta = float(groups[0].get("momentum", 0.9))
    momentum_norm_total = 0.0
    nesterov_norm_total = 0.0
    lr_weighted_total = 0.0
    active = 0
    for group in groups:
        group_beta = float(group.get("momentum", beta))
        if abs(group_beta - beta) > 1e-12:
            raise ValueError("CSD scoring requires a common Muon momentum coefficient")
        group_lr = float(group.get("lr", 0.0))
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None or parameter.ndim < 2:
                continue
            state = optimizer.state.get(parameter, {})
            previous = state.get("momentum_buffer")
            grad32 = gradient.detach().float()
            if torch.is_tensor(previous):
                momentum = previous.detach().float().mul(beta).add(grad32, alpha=1.0-beta)
            else:
                momentum = grad32.mul(1.0-beta)
            direction = torch.lerp(grad32, momentum, beta)
            momentum_norm = torch.linalg.matrix_norm(_muon_matrix_view(momentum), ord="nuc")
            direction_norm = torch.linalg.matrix_norm(_muon_matrix_view(direction), ord="nuc")
            momentum_norm_total += float(momentum_norm.item())
            nesterov_norm_total += float(direction_norm.item())
            lr_weighted_total += group_lr * float(direction_norm.item())
            active += 1
    if active == 0 or not math.isfinite(lr_weighted_total):
        raise RuntimeError("No finite CSD score could be computed from Muon matrix gradients")
    return {
        "score": float(lr_weighted_total),
        "loss": float(loss.detach().float().item()),
        "momentum_dual_norm": float(momentum_norm_total),
        "nesterov_dual_norm": float(nesterov_norm_total),
        "lr_weighted_dual_norm": float(lr_weighted_total),
        "beta": float(beta),
        "active_parameters": float(active),
    }


def capture_rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def snapshot_live_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    optimizer_state = {}
    for name, parameter in parameters.items():
        state = optimizer.state.get(parameter)
        if state:
            optimizer_state[name] = {
                key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in state.items()
            }
    return {
        "training": model.training,
        "params": {name: p.detach().clone() for name, p in parameters.items()},
        "grads": {name: p.grad.detach().clone() if p.grad is not None else None for name, p in parameters.items()},
        "buffers": {name: b.detach().clone() for name, b in buffers.items()},
        "optimizer_state": optimizer_state,
    }


def assert_live_state_unchanged(model: nn.Module, optimizer: torch.optim.Optimizer, snapshot: Dict[str, Any]) -> None:
    if model.training != snapshot["training"]:
        raise RuntimeError("Virtual CSD search changed the live model mode")
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for name, parameter in parameters.items():
        if not torch.equal(parameter.detach(), snapshot["params"][name]):
            raise RuntimeError(f"Virtual CSD search changed live parameter {name}")
        live_grad, old_grad = parameter.grad, snapshot["grads"][name]
        if live_grad is None and old_grad is None:
            pass
        elif live_grad is None or old_grad is None or not torch.equal(live_grad, old_grad):
            raise RuntimeError(f"Virtual CSD search changed live gradient {name}")
        live_state = optimizer.state.get(parameter, {})
        old_state = snapshot["optimizer_state"].get(name, {})
        if set(live_state) != set(old_state):
            raise RuntimeError(f"Virtual CSD search changed optimizer state keys for {name}")
        for key, old_value in old_state.items():
            live_value = live_state[key]
            if torch.is_tensor(old_value):
                if not torch.equal(live_value, old_value):
                    raise RuntimeError(f"Virtual CSD search changed optimizer state {name}.{key}")
            elif live_value != old_value:
                raise RuntimeError(f"Virtual CSD search changed optimizer state {name}.{key}")
    for name, buffer in buffers.items():
        if not torch.equal(buffer, snapshot["buffers"][name]):
            raise RuntimeError(f"Virtual CSD search changed live buffer {name}")


def validate_bounds(low: float, high: float) -> Tuple[float, float]:
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ValueError(f"Teleport bounds must satisfy 0 < low < high; got {low}, {high}.")
    return float(low), float(high)


def generate_tele_scheduler(tele_batch: int, number_of_batch: int, tele_opt: int, tele_cons: int):
    """Use the same scheduled/consecutive teleport convention as the ViT code."""
    if number_of_batch <= 0:
        return []
    tele_batch = min(max(int(tele_batch), 0), int(number_of_batch))
    if tele_opt == 0:
        result = [True] * tele_batch + [False] * (number_of_batch - tele_batch)
        random.shuffle(result)
        return result
    if tele_opt != 1:
        raise ValueError("tele_opt must be 0 (random) or 1 (evenly spaced)")
    result = [False] * number_of_batch
    if tele_batch == 0:
        return result
    tele_cons = max(1, int(tele_cons))
    groups = (tele_batch + tele_cons - 1) // tele_cons
    count = 0
    for group_idx in range(groups):
        start = (group_idx * number_of_batch) // groups
        for offset in range(tele_cons):
            index = start + offset
            if index < number_of_batch and count < tele_batch:
                result[index] = True
                count += 1
    # In edge cases where evenly-spaced groups collide with the end, fill any
    # remaining attempts from the first currently empty slots.
    for index in range(number_of_batch):
        if count >= tele_batch:
            break
        if not result[index]:
            result[index] = True
            count += 1
    return result


def _block_diag(blocks: List[torch.Tensor]) -> torch.Tensor:
    return torch.block_diag(*blocks)


def gen_ape_inv_matrix(
    n_head: int,
    d_head: int,
    tele_high: float,
    tele_low: float,
    tele_sign: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """APE-compatible block-diagonal monomial transform.

    APE imposes no post-projection rotation commutant constraint. We retain the
    repository's established per-head monomial family (permutation plus
    nonzero coordinate scales), repeated over heads. This is a safe subset of
    all invertible per-head APE Q/K transformations.
    """
    low, high = validate_bounds(tele_low, tele_high)
    diagonal = torch.rand(d_head, dtype=torch.float32) * (high - low) + low
    if int(tele_sign) == 1:
        signs = torch.randint(0, 2, (d_head,), dtype=torch.int64).float() * 2.0 - 1.0
        diagonal = diagonal * signs
    base = torch.diag(diagonal)
    row_perm = torch.randperm(d_head)
    col_perm = torch.randperm(d_head)
    matrix = base[row_perm, :][:, col_perm].contiguous()

    inverse = torch.zeros_like(matrix)
    rows, cols = torch.nonzero(matrix, as_tuple=True)
    inverse[cols, rows] = 1.0 / matrix[rows, cols]
    return _block_diag([matrix] * int(n_head)), _block_diag([inverse] * int(n_head))


def gen_rope_inv_matrix(
    n_head: int,
    d_head: int,
    tele_high: float,
    tele_low: float,
    tele_sign: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head scaled 2x2 rotations commuting with adjacent-pair RoPE."""
    low, high = validate_bounds(tele_low, tele_high)
    if d_head % 2:
        raise ValueError(f"RoPE requires even head dimension, got {d_head}")

    matrices, inverses = [], []
    for _ in range(int(n_head)):
        angles = torch.rand(d_head // 2, dtype=torch.float32) * (2.0 * torch.pi)
        scales = torch.rand(d_head // 2, dtype=torch.float32) * (high - low) + low
        if int(tele_sign) == 1:
            signs = torch.randint(0, 2, (d_head // 2,), dtype=torch.int64).float() * 2.0 - 1.0
            scales = scales * signs
        blocks, inverse_blocks = [], []
        for theta, rho in zip(angles, scales):
            c, s = torch.cos(theta), torch.sin(theta)
            blocks.append(torch.stack((torch.stack((rho * c, -rho * s)),
                                       torch.stack((rho * s, rho * c)))))
            inv_rho = 1.0 / rho
            inverse_blocks.append(torch.stack((torch.stack((inv_rho * c, inv_rho * s)),
                                               torch.stack((-inv_rho * s, inv_rho * c)))))
        matrices.append(torch.block_diag(*blocks))
        inverses.append(torch.block_diag(*inverse_blocks))
    return _block_diag(matrices), _block_diag(inverses)


def gen_mlp_permutation(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A permutation and its inverse; exact for any elementwise activation."""
    permutation = torch.eye(dim, dtype=torch.float32)[:, torch.randperm(dim)]
    return permutation, permutation.mT.contiguous()


def attention_blocks(model: nn.Module, tele_layer: str) -> List[nn.Module]:
    layers = list(model.transformer.h)
    if not layers:
        raise ValueError("The GPT-2 model has no transformer blocks")
    if tele_layer == "all":
        return layers
    if tele_layer == "first":
        return [layers[0]]
    if tele_layer == "last":
        return [layers[-1]]
    raise ValueError(f"Unsupported tele_layer: {tele_layer}")


def qkv_modules(model: nn.Module, tele_layer: str) -> List[nn.Linear]:
    return [block.attn.c_attn for block in attention_blocks(model, tele_layer)]


def _matrix_scale_summary(matrix: Optional[torch.Tensor]) -> List[float]:
    if matrix is None:
        return []
    return torch.linalg.vector_norm(matrix.float(), dim=1).detach().cpu().tolist()


def generate_symmetry_candidates(
    model: nn.Module,
    args: Any,
    low: float,
    high: float,
    count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    validate_bounds(low, high)
    use_attention = bool(getattr(args, "tele_att", 1))
    use_mlp = bool(getattr(args, "tele_mlp", 0))
    if not use_attention and not use_mlp:
        raise ValueError("Enable --tele-att 1 and/or --tele-mlp 1")

    config = model.config
    n_head = int(config.n_head)
    d_head = int(config.n_embd // config.n_head)
    mlp_dim = int(model.transformer.h[0].mlp.c_fc.out_features)
    use_rope = model.position_embedding == "rope"

    original_rng = capture_rng_state()
    candidates: List[Dict[str, Any]] = []
    try:
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
                    matrix = torch.eye(config.n_embd, dtype=torch.float32)
                    inverse = matrix.clone()
                elif use_rope:
                    matrix, inverse = gen_rope_inv_matrix(
                        n_head, d_head, high, low, int(getattr(args, "tele_sign", 0))
                    )
                else:
                    matrix, inverse = gen_ape_inv_matrix(
                        n_head, d_head, high, low, int(getattr(args, "tele_sign", 0))
                    )
                candidate["M_att"], candidate["M_att_inv"] = matrix, inverse

            if use_mlp:
                if index == 0:
                    matrix = torch.eye(mlp_dim, dtype=torch.float32)
                    inverse = matrix.clone()
                else:
                    # GPT-2 uses GELU. Only permutations commute with GELU;
                    # positive coordinate scaling is not a GELU symmetry.
                    matrix, inverse = gen_mlp_permutation(mlp_dim)
                candidate["M_mlp"], candidate["M_mlp_inv"] = matrix, inverse

            candidate["attention_scale_summary"] = _matrix_scale_summary(candidate["M_att"])
            candidate["mlp_scale_summary"] = _matrix_scale_summary(candidate["M_mlp"])
            candidate["scale_summary"] = (
                candidate["attention_scale_summary"]
                if use_attention else candidate["mlp_scale_summary"]
            )
            scales = candidate["attention_scale_summary"] + candidate["mlp_scale_summary"]
            candidate["scale_min"] = min(scales) if scales else 1.0
            candidate["scale_max"] = max(scales) if scales else 1.0
            candidate["transform_type"] = "gpt2_rope_block" if use_rope else "gpt2_ape_monomial"
            candidates.append(candidate)
    finally:
        restore_rng_state(original_rng)

    if not candidates or not candidates[0]["identity"]:
        raise ValueError("At least one candidate (identity) is required")
    return candidates


def teleportation_att(model: nn.Module, matrix: torch.Tensor, matrix_inv: torch.Tensor, tele_layer: str) -> None:
    """Apply Q'=M Q, K'=M^{-T}K to fused GPT-2 QKV linear projections."""
    hidden_size = int(model.config.n_embd)
    for block in attention_blocks(model, tele_layer):
        projection = block.attn.c_attn
        transform = matrix.to(device=projection.weight.device, dtype=projection.weight.dtype)
        inverse = matrix_inv.to(device=projection.weight.device, dtype=projection.weight.dtype)
        with torch.no_grad():
            q_weight = projection.weight[:hidden_size].clone()
            k_weight = projection.weight[hidden_size : 2 * hidden_size].clone()
            projection.weight[:hidden_size].copy_(transform @ q_weight)
            projection.weight[hidden_size : 2 * hidden_size].copy_(inverse.mT @ k_weight)
            if projection.bias is not None:
                q_bias = projection.bias[:hidden_size].clone()
                k_bias = projection.bias[hidden_size : 2 * hidden_size].clone()
                projection.bias[:hidden_size].copy_(transform @ q_bias)
                projection.bias[hidden_size : 2 * hidden_size].copy_(inverse.mT @ k_bias)
            # V weight and bias are unchanged.


def teleportation_mlp(model: nn.Module, matrix: torch.Tensor, matrix_inv: torch.Tensor) -> None:
    """Apply a GELU-preserving hidden-unit permutation to every GPT-2 block."""
    for block in model.transformer.h:
        fc = block.mlp.c_fc
        proj = block.mlp.c_proj
        transform = matrix.to(device=fc.weight.device, dtype=fc.weight.dtype)
        inverse = matrix_inv.to(device=proj.weight.device, dtype=proj.weight.dtype)
        with torch.no_grad():
            fc.weight.copy_(transform @ fc.weight)
            if fc.bias is not None:
                fc.bias.copy_(transform @ fc.bias)
            proj.weight.copy_(proj.weight @ inverse)
            # c_proj.bias is in the residual stream and is unchanged.


def apply_symmetry_candidate(model: nn.Module, candidate: Dict[str, Any], args: Any) -> None:
    if bool(getattr(args, "tele_att", 1)):
        teleportation_att(model, candidate["M_att"], candidate["M_att_inv"], args.tele_layer)
    if bool(getattr(args, "tele_mlp", 0)):
        teleportation_mlp(model, candidate["M_mlp"], candidate["M_mlp_inv"])


def create_lm_muon_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
    adam_lr: float = 5e-5,
    adam_weight_decay: Optional[float] = None,
    nesterov: bool = False,
) -> torch.optim.Optimizer:
    """Muon for GPT-2 block matrices; AdamW for embeddings, biases and norms."""
    muon_parameters: List[nn.Parameter] = []
    adam_parameters: List[nn.Parameter] = []
    muon_suffixes = (
        ".attn.c_attn.weight",
        ".attn.c_proj.weight",
        ".mlp.c_fc.weight",
        ".mlp.c_proj.weight",
    )
    for name, parameter in model.named_parameters():
        is_block_matrix = parameter.ndim == 2 and name.startswith("transformer.h.") and name.endswith(muon_suffixes)
        (muon_parameters if is_block_matrix else adam_parameters).append(parameter)
    if not muon_parameters:
        raise ValueError("No GPT-2 transformer matrices were assigned to Muon")
    if not adam_parameters:
        raise ValueError("No GPT-2 embedding/bias/normalization parameters were assigned to AdamW")

    optimizer = SingleDeviceMuonWithAuxAdam(
        [
            {
                "params": muon_parameters,
                "use_muon": True,
                "lr": float(lr),
                "momentum": float(momentum),
                "nesterov": bool(nesterov),
                "weight_decay": float(weight_decay),
            },
            {
                "params": adam_parameters,
                "use_muon": False,
                "lr": float(adam_lr),
                "betas": (0.9, 0.95),
                "eps": 1e-10,
                "weight_decay": float(weight_decay if adam_weight_decay is None else adam_weight_decay),
            },
        ]
    )
    # Keep readable names outside optimizer state_dicts; used only for W&B
    # diagnostics and reconstructed automatically for cloned optimizers.
    optimizer.parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    return optimizer


def transport_teleportation_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    candidate: Dict[str, Any],
    args: Any,
) -> None:
    """Transport Muon buffers and reset affected elementwise Adam states.

    The fused QKV momentum buffer has row blocks [Q;K;V]. For the parameter
    transformations W_Q'=M W_Q and W_K'=M^{-T}W_K, its gradient-like momentum
    transforms as B_Q'=M^{-T}B_Q and B_K'=M B_K. V is unchanged.
    """
    if bool(candidate.get("identity", False)):
        return

    hidden_size = int(model.config.n_embd)
    if bool(getattr(args, "tele_att", 1)):
        matrix = candidate["M_att"]
        matrix_inv = candidate["M_att_inv"]
        for block in attention_blocks(model, args.tele_layer):
            projection = block.attn.c_attn
            state = optimizer.state.get(projection.weight, {})
            momentum = state.get("momentum_buffer")
            if torch.is_tensor(momentum):
                transform = matrix.to(device=momentum.device, dtype=momentum.dtype)
                inverse = matrix_inv.to(device=momentum.device, dtype=momentum.dtype)
                with torch.no_grad():
                    q_buffer = momentum[:hidden_size].clone()
                    k_buffer = momentum[hidden_size : 2 * hidden_size].clone()
                    momentum[:hidden_size].copy_(inverse.mT @ q_buffer)
                    momentum[hidden_size : 2 * hidden_size].copy_(transform @ k_buffer)
            # Q/K bias coordinates changed but are slices of one Adam tensor;
            # reset that state rather than retaining stale elementwise moments.
            if projection.bias is not None:
                optimizer.state.pop(projection.bias, None)

    if bool(getattr(args, "tele_mlp", 0)):
        matrix = candidate["M_mlp"]
        matrix_inv = candidate["M_mlp_inv"]
        for block in model.transformer.h:
            fc = block.mlp.c_fc
            proj = block.mlp.c_proj
            fc_state = optimizer.state.get(fc.weight, {})
            fc_momentum = fc_state.get("momentum_buffer")
            if torch.is_tensor(fc_momentum):
                inverse = matrix_inv.to(device=fc_momentum.device, dtype=fc_momentum.dtype)
                with torch.no_grad():
                    fc_momentum.copy_(inverse.mT @ fc_momentum)
            proj_state = optimizer.state.get(proj.weight, {})
            proj_momentum = proj_state.get("momentum_buffer")
            if torch.is_tensor(proj_momentum):
                transform = matrix.to(device=proj_momentum.device, dtype=proj_momentum.dtype)
                with torch.no_grad():
                    proj_momentum.copy_(proj_momentum @ transform.mT)
            if fc.bias is not None:
                optimizer.state.pop(fc.bias, None)


def _clone_optimizer_for_model(
    model_clone: nn.Module, optimizer: torch.optim.Optimizer
) -> torch.optim.Optimizer:
    muon_group = next(group for group in optimizer.param_groups if group.get("use_muon", False))
    adam_group = next(group for group in optimizer.param_groups if not group.get("use_muon", False))
    return create_lm_muon_optimizer(
        model_clone,
        lr=float(muon_group["lr"]),
        weight_decay=float(muon_group.get("weight_decay", 0.0)),
        momentum=float(muon_group.get("momentum", 0.9)),
        adam_lr=float(adam_group["lr"]),
        nesterov=bool(muon_group.get("nesterov", False)),
        adam_weight_decay=float(adam_group.get("weight_decay", 0.0)),
    )


def clone_model_and_optimizer(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[nn.Module, torch.optim.Optimizer]:
    model_clone = copy.deepcopy(model)
    optimizer_clone = _clone_optimizer_for_model(model_clone, optimizer)
    real_named = dict(model.named_parameters())
    clone_named = dict(model_clone.named_parameters())
    for name, parameter in real_named.items():
        state = optimizer.state.get(parameter)
        if not state:
            continue
        optimizer_clone.state[clone_named[name]] = {
            key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in state.items()
        }
    return model_clone, optimizer_clone


def _loss_for_batch(model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor], criterion, args):
    inputs, targets = batch
    amp_dtype = getattr(args, "amp_dtype", "none")
    device_type = inputs.device.type
    if device_type == "cuda" and amp_dtype == "bfloat16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inputs).logits
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    else:
        logits = model(input_ids=inputs).logits
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return loss


def try_muon_alignment_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Backward-compatible alias for Nesterov CSD-L2 teleportation."""
    try:
        from .csd_nesterov import try_csd_l2_nesterov_teleportation
    except ImportError:  # python src/lm/train.py
        from csd_nesterov import try_csd_l2_nesterov_teleportation
    return try_csd_l2_nesterov_teleportation(
        model, optimizer, batch, criterion, args, step_index
    )


def try_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Backward-compatible dispatcher to the dedicated LM selection module."""
    try:
        from .csd_nesterov import try_teleportation as dispatch
    except ImportError:  # python src/lm/train.py
        from csd_nesterov import try_teleportation as dispatch
    return dispatch(model, optimizer, batch, criterion, args, step_index)


__all__ = [
    "create_lm_muon_optimizer",
    "generate_tele_scheduler",
    "generate_symmetry_candidates",
    "teleportation_att",
    "teleportation_mlp",
    "transport_teleportation_optimizer_state",
    "try_muon_alignment_teleportation",
    "try_teleportation",
]
