import copy
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from muon import _muon_update, create_muon_optimizer


def calculate_grad_L2(net):
    total_L2 = 0.0
    for name, param in net.named_parameters():
        if param.grad is not None:
            total_L2 += param.grad.norm(2).item() ** 2
    return total_L2 ** 0.5


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
    grad_l2_base = calculate_grad_L2(base_model)
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
        grad_l2_i = calculate_grad_L2(model_i)
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


def compute_symmetry_muon_alignment(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
) -> float:
    """Sum gradient/update alignment over all transformed Muon weight matrices."""
    if bool(getattr(args, "tele_att", 1)) and not bool(getattr(args, "tele_mlp", 0)):
        # Preserve compatibility with callers/tests that patch the existing Q/K
        # scorer, as well as its exact Q/K-only semantics.
        return compute_qk_muon_alignment(model, optimizer, batch, criterion, args.tele_layer)

    optimizer.zero_grad(set_to_none=True)
    logits = model(pixel_values=batch[0]).logits
    loss = criterion(logits, batch[1])
    loss.backward()

    if bool(getattr(args, "tele_att", 1)):
        parameters = qk_weight_parameters(model, args.tele_layer)
    else:
        parameters = []
    if bool(getattr(args, "tele_mlp", 0)):
        parameters.extend(mlp_weight_parameters(model))

    score = 0.0
    seen = set()
    for parameter in parameters:
        if id(parameter) in seen or parameter.grad is None:
            continue
        seen.add(id(parameter))
        group = parameter_group(optimizer, parameter)
        if group is None or not group.get("use_muon", False):
            continue
        update = reset_muon_update(parameter.grad, momentum=group.get("momentum", 0.95))
        grad = parameter.grad.detach().float().reshape_as(update)
        score += float(group["lr"]) * float(torch.sum(grad * update.float()).item())
    return score


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


def try_muon_alignment_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Select and apply the symmetry candidate maximizing Muon alignment.

    This path deliberately does not evaluate virtual post-update losses. Every
    candidate is scored by S_t(g) = sum_j lr_j <G_j(g), O_j(g)>_F, where O_j
    is computed by the repository's actual Muon update/orthogonalization code
    with a fresh momentum buffer. The highest-scoring candidate is selected
    directly, including the identity candidate.

    Public signature and telemetry keys are retained for compatibility. Loss-
    based telemetry fields are set to None because no such losses are computed.
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

    was_training = model.training
    committed = False
    accepted = False
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
            align_model, align_optimizer = clone_model_and_optimizer(model, optimizer)
            # Give every candidate the same stochastic starting point.
            restore_rng_state(rng_state)
            apply_symmetry_candidate(align_model, candidate, args)
            alignment = compute_symmetry_muon_alignment(
                align_model, align_optimizer, batch, criterion, args
            )
            if not math.isfinite(float(alignment)):
                raise ValueError(
                    f"Non-finite Muon alignment score for candidate {index}: {alignment}."
                )

            results.append(
                {
                    "candidate_index": index,
                    "identity": bool(candidate["identity"]),
                    # Preserve the former telemetry schema while reporting the
                    # coordinate scale magnitudes implied by the full matrices.
                    "scale": candidate["scale_summary"],
                    "scale_min": float(candidate["scale_min"]),
                    "scale_max": float(candidate["scale_max"]),
                    "attention_scale": candidate["attention_scale_summary"],
                    "mlp_scale": candidate["mlp_scale_summary"],
                    "transform_type": "mnistcifar_symmetry_preserving_matrix",
                    "uses_attention": bool(getattr(args, "tele_att", 1)),
                    "uses_mlp": bool(getattr(args, "tele_mlp", 0)),
                    "alignment": float(alignment),
                    "alignment_evaluated": True,
                    "alignment_rank": None,
                    "virtual_loss_evaluated": False,
                    "post_step_loss": None,
                    "searched": False,
                }
            )

        # Exact discrete argmax over the candidate set. The identity candidate
        # participates in the same objective; in ties, the earlier candidate
        # (identity first) wins deterministically.
        ranked = sorted(
            results,
            key=lambda item: (-float(item["alignment"]), int(item["candidate_index"])),
        )
        for rank, item in enumerate(ranked, start=1):
            item["alignment_rank"] = rank
        selected = ranked[0]
        selected_index = int(selected["candidate_index"])
        selected_candidate = candidates[selected_index]
        identity_result = next(item for item in results if item["identity"])
        non_identity_results = [item for item in results if not item["identity"]]
        best_non_identity = max(
            non_identity_results,
            key=lambda item: float(item["alignment"]),
        ) if non_identity_results else None

        # Verify scoring was read-only before committing the selected transform.
        restore_rng_state(rng_state)
        model.train(was_training)
        assert_live_state_unchanged(model, optimizer, live_snapshot)

        # Algorithm 8 resets the optimizer state for the parameter families in
        # the selected symmetry. This also applies when identity wins: in that
        # case parameters stay the same, but the reset-only branch is selected.
        apply_symmetry_candidate(model, selected_candidate, args)
        reset_teleportation_optimizer_state(model, optimizer, args)
        committed = True
        accepted = not bool(selected["identity"])
        reason = (
            "selected_max_alignment"
            if accepted
            else "identity_selected_max_alignment_state_reset"
        )

        identity_alignment = float(identity_result["alignment"])
        selected_alignment = float(selected["alignment"])
        alignment_gain = selected_alignment - identity_alignment
        search_time = round(time.perf_counter() - started, 6)

        telemetry = {
            "method": "muon_alignment",
            "accepted": accepted,
            "reason": reason,
            "step": step_index,
            "search_time_seconds": search_time,
            # Kept for log/schema compatibility; these objectives are no longer
            # evaluated by the alignment-only selection algorithm.
            "regular_muon_post_step_loss": None,
            "identity_reset_loss": None,
            "identity_alignment": identity_alignment,
            "best_non_identity_candidate_loss": None,
            "beats_regular_baseline": None,
            "beats_reset_only": None,
            "accepted_beyond_reset_only": None,
            "selected_alignment": selected_alignment,
            "selected_post_step_loss": None,
            "selected_identity": bool(selected["identity"]),
            "selected_candidate_index": selected_index,
            "selected_scale": selected["scale"],
            "scale_min": selected["scale_min"],
            "scale_max": selected["scale_max"],
            "selected_attention_scale": selected_candidate["attention_scale_summary"],
            "selected_mlp_scale": selected_candidate["mlp_scale_summary"],
            "identity_rank": int(identity_result["alignment_rank"]),
            "state_reset": True,
            "tele_layer": args.tele_layer,
            "transform_family": "mnistcifar_symmetry_preserving_matrix",
            "uses_attention": bool(getattr(args, "tele_att", 1)),
            "uses_mlp": bool(getattr(args, "tele_mlp", 0)),
            "alignment_gain": alignment_gain,
            "post_loss_gain": None,
            "proposed_candidate_index": (
                int(best_non_identity["candidate_index"])
                if best_non_identity is not None else None
            ),
            "proposed_post_loss_gain": None,
            "accepted_candidate_index": selected_index if accepted else None,
            "accepted_post_loss_gain": None,
            # No loss-based shortlist remains; all candidates were scored.
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
    if method == "muon_alignment":
        return try_muon_alignment_teleportation(model, optimizer, batch, criterion, args, step_index)
    raise ValueError(f"Unsupported tele_method: {method}")
