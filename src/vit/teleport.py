import copy
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from muon import _muon_update, create_muon_optimizer
from criteria import get_scoring_criterion
from criteria.naive import gradient_l2_score


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


def try_naive_teleportation(vit_model, samples, targets, criterion, args, high, low, sign=0):
    vit_model.eval()
    base_model = copy.deepcopy(vit_model)
    base_outputs = base_model(pixel_values=samples).logits
    base_loss = criterion(base_outputs, targets)
    base_loss.backward()
    grad_l2_base = gradient_l2_score(base_model)
    max_grad_l2 = grad_l2_base
    max_M_att = None
    max_M_mlp = None
    max_M_att_inv = None
    max_M_mlp_inv = None
    count_large_grad = 0
    for _ in range(args.n_teleport - 1):
        model_i = copy.deepcopy(vit_model)
        if args.tele_att:
            if args.position_embedding == "rope":
                M_att, M_att_inv = gen_rope_inv_matrix(
                    n_head=args.num_heads,
                    d_head=args.d_model // args.num_heads,
                    tele_high=high,
                    tele_low=low,
                    tele_sign=sign,
                )
            else:
                M_att, M_att_inv = gen_inv_matrix(
                    n_head=args.num_heads,
                    d_head=args.d_model // args.num_heads,
                    tele_high=high,
                    tele_low=low,
                    tele_sign=sign,
                )
            teleportation_att(model_i, M_att, M_att_inv, args.tele_layer)
        if args.tele_mlp:
            M_mlp, M_mlp_inv = gen_inv_matrix(
                n_head=1,
                d_head=args.intermediate_size,
                tele_high=high,
                tele_low=low,
                # The repo ViT uses ReLU, which preserves positive monomial
                # hidden-unit changes but not arbitrary sign flips.
                tele_sign=0,
            )
            teleportation_mlp(model_i, M_mlp, M_mlp_inv)
        outputs_i = model_i(pixel_values=samples).logits
        loss_i = criterion(outputs_i, targets)
        loss_i.backward()
        grad_l2_i = gradient_l2_score(model_i)
        if grad_l2_i > grad_l2_base:
            count_large_grad += 1
        if grad_l2_i > max_grad_l2:
            max_grad_l2 = grad_l2_i
            if args.tele_att:
                max_M_att = M_att
                max_M_att_inv = M_att_inv
            if args.tele_mlp:
                max_M_mlp = M_mlp
                max_M_mlp_inv = M_mlp_inv

    if count_large_grad > args.n_teleport / 2:
        if max_M_att is not None and max_M_att_inv is not None:
            teleportation_att(vit_model, max_M_att, max_M_att_inv, args.tele_layer)
        if max_M_mlp is not None and max_M_mlp_inv is not None:
            teleportation_mlp(vit_model, max_M_mlp, max_M_mlp_inv)
        vit_model.train()
        return True
    vit_model.train()
    return False


def validate_bounds(low: float, high: float) -> Tuple[float, float]:
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ValueError(f"Teleport bounds must satisfy 0 < low < high; got low={low}, high={high}.")
    return low, high


def identity_scale(dim: int) -> torch.Tensor:
    return torch.ones(dim, dtype=torch.float32)


def validate_qk_scale(scale: torch.Tensor, low: float, high: float, rope: bool) -> torch.Tensor:
    if scale.ndim != 1:
        raise ValueError(f"Q/K scale must be a 1D tensor, got shape {tuple(scale.shape)}")
    if not torch.isfinite(scale).all():
        raise ValueError("Q/K scale must be finite.")
    if torch.any(scale <= 0):
        raise ValueError("Q/K scale must be strictly positive.")
    identity = torch.ones_like(scale)
    if not torch.equal(scale, identity) and (torch.any(scale < low) or torch.any(scale > high)):
        raise ValueError(f"Q/K scale must lie in [{low}, {high}].")
    if rope:
        if scale.numel() % 2 != 0:
            raise ValueError("RoPE Q/K scales require an even number of coordinates.")
        even = scale[0::2]
        odd = scale[1::2]
        if not torch.allclose(even, odd):
            raise ValueError("RoPE Q/K scales must be tied within each adjacent rotary pair.")
    return scale


def candidate_qk_scales(
    dim: int,
    low: float,
    high: float,
    count: int,
    seed: int,
    rope: bool,
) -> List[torch.Tensor]:
    validate_bounds(low, high)
    rng = random.Random(seed)
    candidates: List[torch.Tensor] = [identity_scale(dim)]
    pair_count = dim // 2
    remainder = dim % 2
    for _ in range(max(0, count - 1)):
        values: List[float] = []
        if rope:
            for _pair in range(pair_count):
                scale = math.exp(rng.uniform(math.log(low), math.log(high)))
                scale = min(high, max(low, scale))
                values.extend([scale, scale])
            if remainder:
                extra = math.exp(rng.uniform(math.log(low), math.log(high)))
                values.append(min(high, max(low, extra)))
        else:
            for _idx in range(dim):
                scale = math.exp(rng.uniform(math.log(low), math.log(high)))
                values.append(min(high, max(low, scale)))
        candidate = torch.tensor(values, dtype=torch.float32)
        validate_qk_scale(candidate, low, high, rope)
        candidates.append(candidate)
    return candidates


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


def qk_weight_parameters(model: nn.Module, tele_layer: str) -> List[nn.Parameter]:
    weights = []
    for query, key in qk_linear_modules(model, tele_layer):
        weights.append(query.weight)
        weights.append(key.weight)
    return weights


def apply_qk_diagonal_transform(model: nn.Module, scale: torch.Tensor, tele_layer: str, rope: bool, low: float, high: float) -> None:
    scale = validate_qk_scale(scale.detach().clone().float().cpu(), low, high, rope)
    for query, key in qk_linear_modules(model, tele_layer):
        diag = scale.to(device=query.weight.device, dtype=query.weight.dtype)
        inv = torch.reciprocal(diag)
        if diag.numel() != query.weight.shape[0] or diag.numel() != key.weight.shape[0]:
            raise ValueError("Q/K scale length must match projection out_features.")
        with torch.no_grad():
            query.weight.mul_(diag.unsqueeze(1))
            key.weight.mul_(inv.unsqueeze(1))
            if query.bias is not None:
                query.bias.mul_(diag)
            if key.bias is not None:
                key.bias.mul_(inv)


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
        raise ValueError("csd_l2 requires --tele-att 1, --tele-mlp 1, or both.")

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


def mlp_weight_parameters(model: nn.Module) -> List[nn.Parameter]:
    weights: List[nn.Parameter] = []
    for block in model.vit.encoder.layer:
        weights.append(block.intermediate.dense.weight)
        weights.append(block.output.dense.weight)
    return weights


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


def reset_teleportation_optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer, args: Any) -> None:
    for _, parameter in teleportation_parameter_items(model, args):
        optimizer.state.pop(parameter, None)


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


def reset_qk_optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer, tele_layer: str) -> None:
    for _, parameter in qk_parameter_items(model, tele_layer):
        optimizer.state.pop(parameter, None)


def parameter_group(optimizer: torch.optim.Optimizer, parameter: nn.Parameter) -> Optional[dict]:
    for group in optimizer.param_groups:
        for group_parameter in group["params"]:
            if group_parameter is parameter:
                return group
    return None


def reset_muon_update(gradient: torch.Tensor, momentum: float) -> torch.Tensor:
    momentum_buffer = torch.zeros_like(gradient)
    grad = gradient.detach().clone()
    return _muon_update(grad, momentum_buffer, beta=momentum)


def compute_qk_muon_alignment(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    tele_layer: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(pixel_values=batch[0]).logits
    loss = criterion(logits, batch[1])
    loss.backward()

    score = 0.0
    for parameter in qk_weight_parameters(model, tele_layer):
        group = parameter_group(optimizer, parameter)
        if group is None or not group.get("use_muon", False) or parameter.grad is None:
            continue
        update = reset_muon_update(parameter.grad, momentum=group.get("momentum", 0.95))
        grad = parameter.grad.detach().float().reshape_as(update)
        score += float(group["lr"]) * float(torch.sum(grad * update.float()).item())
    return score


def virtual_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(pixel_values=batch[0]).logits
    loss = criterion(logits, batch[1])
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        post_loss = criterion(model(pixel_values=batch[0]).logits, batch[1])
    return float(post_loss.item())


def _candidate_sort_key(result: Dict[str, Any]) -> Tuple[float, int]:
    return (float(result["alignment"]), -1 if result["identity"] else 0)


def _resolve_alignment_shortlist(args: Any, non_identity_count: int) -> int:
    raw_shortlist = getattr(args, "tele_align_shortlist", None)
    if raw_shortlist is None:
        raw_shortlist = min(3, max(1, non_identity_count))
    shortlist = int(raw_shortlist)
    if shortlist <= 0:
        raise ValueError("--tele-align-shortlist must be positive.")
    if non_identity_count == 0:
        return 0
    return min(shortlist, non_identity_count)


def _try_csd_l2_teleportation_impl(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
    metrics_fn,
    method_name: str,
    score_definition: str,
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate candidates with one scoring criterion and commit the winner.

    Candidate creation, model/optimizer cloning, transformations, state
    transport, and commitment live here. A criterion only computes a score
    from the candidate optimizer after its gradients have been calculated.
    """
    started = time.perf_counter()
    rng_state = capture_rng_state()
    live_snapshot = snapshot_live_state(model, optimizer)
    low, high = validate_bounds(args.tele_low, args.tele_high)
    candidate_count = int(args.tele_candidates)
    if candidate_count < 2:
        raise ValueError(
            "Scored teleportation requires at least one non-identity candidate; "
            "--tele-candidates must be >= 2."
        )

    if not any(bool(group.get("use_muon", False)) for group in optimizer.param_groups):
        raise ValueError("Scored teleportation requires a Muon optimizer parameter group.")

    was_training = model.training
    committed = False
    try:
        model.train()
        candidates = generate_symmetry_candidates(
            model=model,
            args=args,
            low=low,
            high=high,
            count=candidate_count,
            seed=args.seed + step_index,
        )

        results: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            candidate_model, candidate_optimizer = clone_model_and_optimizer(model, optimizer)
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
            metrics = metrics_fn(candidate_optimizer)
            score = float(metrics["score"])
            if not math.isfinite(score):
                raise ValueError(
                    f"Non-finite {method_name} score for candidate {index}: {score}."
                )

            results.append({
                "candidate_index": index,
                "identity": bool(candidate["identity"]),
                "score": score,
                "squared_score": float(metrics["squared_score"]),
                "norm_kind": metrics.get("norm_kind", "nuclear"),
                "matrix_count": int(metrics["matrix_count"]),
                "matrix_norms": metrics.get("matrix_norms", []),
                "momentum_direction": metrics.get("momentum_direction", "unknown"),
                "scale": candidate["scale_summary"],
                "scale_min": float(candidate["scale_min"]),
                "scale_max": float(candidate["scale_max"]),
                "attention_scale": candidate["attention_scale_summary"],
                "mlp_scale": candidate["mlp_scale_summary"],
            })

        # Identity is candidate zero and wins exact ties.
        ranked = sorted(
            results, key=lambda item: (-item["score"], int(item["candidate_index"]))
        )
        selected = ranked[0]
        selected_index = int(selected["candidate_index"])
        selected_candidate = candidates[selected_index]
        identity_result = next(item for item in results if item["identity"])

        # Virtual evaluation must not mutate the live model, gradients, or state.
        restore_rng_state(rng_state)
        model.train(was_training)
        assert_live_state_unchanged(model, optimizer, live_snapshot)

        accepted = not bool(selected["identity"])
        if accepted:
            apply_symmetry_candidate(model, selected_candidate, args)
            transport_teleportation_optimizer_state(
                model, optimizer, selected_candidate, args
            )
            committed = True
            reason = f"selected_max_{method_name}_score"
        else:
            reason = f"identity_selected_max_{method_name}_score_noop"

        identity_score = float(identity_result["score"])
        selected_score = float(selected["score"])
        score_gain = selected_score - identity_score
        return accepted, {
            "method": method_name,
            "accepted": accepted,
            "reason": reason,
            "step": step_index,
            "search_time_seconds": round(time.perf_counter() - started, 6),
            "score_definition": score_definition,
            "score_norm_kind": selected["norm_kind"],
            "momentum_direction": selected["momentum_direction"],
            "identity_score": identity_score,
            "selected_score": selected_score,
            "score_gain": score_gain,
            "selected_csd_l2_score": selected_score,
            "csd_l2_gain": score_gain,
            "selected_candidate_index": selected_index,
            "selected_identity": bool(selected["identity"]),
            "selected_scale": selected["scale"],
            "selected_scale_min": selected["scale_min"],
            "selected_scale_max": selected["scale_max"],
            "num_candidates_evaluated": len(results),
            "momentum_state_transported": accepted,
            "muon_momentum_reset": False,
            "auxiliary_adam_state_reset": accepted,
            "results": results,
        }
    finally:
        restore_rng_state(rng_state)
        model.train(was_training)
        if not committed:
            assert_live_state_unchanged(model, optimizer, live_snapshot)


def try_scored_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
    method_name: str,
) -> Tuple[bool, Dict[str, Any]]:
    """Run the common candidate search with a criterion from ``criteria/``."""
    criterion_spec = get_scoring_criterion(method_name)
    return _try_csd_l2_teleportation_impl(
        model, optimizer, batch, criterion, args, step_index,
        metrics_fn=criterion_spec["metrics_fn"],
        method_name=criterion_spec["method_name"],
        score_definition=criterion_spec["score_definition"],
    )


def try_csd_l2_teleportation(model, optimizer, batch, criterion, args, step_index):
    return try_scored_teleportation(
        model, optimizer, batch, criterion, args, step_index, "csd_l2"
    )


def try_csd_l2_nesterov_teleportation(model, optimizer, batch, criterion, args, step_index):
    return try_scored_teleportation(
        model, optimizer, batch, criterion, args, step_index, "csd_l2_nesterov"
    )


def try_csd_l2_nesterov_frobenius_teleportation(model, optimizer, batch, criterion, args, step_index):
    return try_scored_teleportation(
        model, optimizer, batch, criterion, args, step_index, "csd_l2_nesterov_frobenius"
    )


def try_csd_l2_nesterov_post_orth_teleportation(model, optimizer, batch, criterion, args, step_index):
    return try_scored_teleportation(
        model, optimizer, batch, criterion, args, step_index, "csd_l2_nesterov_post_orth"
    )
def try_muon_alignment_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    return try_csd_l2_teleportation(model, optimizer, batch, criterion, args, step_index)


def try_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Dispatch teleportation; candidate generation stays in this module."""
    method = getattr(args, "tele_method", "none")
    if method in {"none", None}:
        return False, {"method": "none", "accepted": False, "step": step_index}
    if method == "naive":
        accepted = try_naive_teleportation(
            vit_model=model,
            samples=batch[0],
            targets=batch[1],
            criterion=criterion,
            args=args,
            high=args.tele_high,
            low=args.tele_low,
            sign=args.tele_sign,
        )
        return accepted, {"method": "naive", "accepted": accepted, "step": step_index}
    if method in {
        "csd_l2",
        "muon_alignment",
        "csd_l2_nesterov",
        "csd_l2_nesterov_frobenius",
        "csd_l2_nesterov_post_orth",
        "csd_l2_nesterov_post_orth_frobenius",
    }:
        return try_scored_teleportation(
            model, optimizer, batch, criterion, args, step_index, method
        )
    raise ValueError(f"Unsupported tele_method: {method}")
