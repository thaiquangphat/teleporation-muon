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
    layers = vit_model.vit.encoder.layer

    if tele_layer == "all":
        for i, block in enumerate(layers):
            q_linear = block.attention.attention.query
            k_linear = block.attention.attention.key
            with torch.no_grad():
                M = M.to(q_linear.weight.device)
                M_inv = M_inv.to(k_linear.weight.device)
                q_linear.weight.data = torch.matmul(M, q_linear.weight.data)
                q_linear.bias.data = torch.matmul(q_linear.bias.data, torch.transpose(M, -1, -2))
                k_linear.weight.data = torch.matmul(torch.transpose(M_inv, -1, -2), k_linear.weight.data)
                k_linear.bias.data = torch.matmul(k_linear.bias.data, M_inv)
    else:
        if tele_layer == "first":
            block = layers[0]
        else:
            block = layers[-1]

        q_linear = block.attention.attention.query
        k_linear = block.attention.attention.key
        with torch.no_grad():
            M = M.to(q_linear.weight.device)
            M_inv = M_inv.to(k_linear.weight.device)
            q_linear.weight.data = torch.matmul(M, q_linear.weight.data)
            q_linear.bias.data = torch.matmul(q_linear.bias.data, torch.transpose(M, -1, -2))
            k_linear.weight.data = torch.matmul(torch.transpose(M_inv, -1, -2), k_linear.weight.data)
            k_linear.bias.data = torch.matmul(k_linear.bias.data, M_inv)


def teleportation_mlp(vit_model, M, M_inv):
    layers = vit_model.vit.encoder.layer
    for i, block in enumerate(layers):
        intermediate_linear = block.intermediate.dense
        output_linear = block.output.dense
        with torch.no_grad():
            M = M.to(intermediate_linear.weight.device)
            M_inv = M_inv.to(output_linear.weight.device)
            intermediate_linear.weight.data = torch.matmul(M, intermediate_linear.weight.data)
            intermediate_linear.bias.data = torch.matmul(intermediate_linear.bias.data, torch.transpose(M, -1, -2))
            output_linear.weight.data = torch.matmul(output_linear.weight.data, M_inv)


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
                tele_sign=sign,
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
    started = time.perf_counter()
    rng_state = capture_rng_state()
    live_snapshot = snapshot_live_state(model, optimizer)
    low, high = validate_bounds(args.tele_low, args.tele_high)
    rope = args.position_embedding == "rope"
    query = qk_linear_modules(model, args.tele_layer)[0][0]
    dim = query.out_features
    if args.tele_candidates < 2:
        raise ValueError("muon_alignment requires at least one non-identity candidate; --tele-candidates must be >= 2.")

    was_training = model.training
    accepted = False
    model.train()
    try:
        candidates = candidate_qk_scales(
            dim=dim,
            low=low,
            high=high,
            count=args.tele_candidates,
            seed=args.seed + step_index,
            rope=rope,
        )
        shortlist_size = _resolve_alignment_shortlist(args, max(0, args.tele_candidates - 1))
        shortlisted_indices = set()

        regular_model, regular_optimizer = clone_model_and_optimizer(model, optimizer)
        restore_rng_state(rng_state)
        regular_loss = virtual_optimizer_step(regular_model, regular_optimizer, batch, criterion)

        results: List[Dict[str, Any]] = []
        for index, scale in enumerate(candidates):
            identity = index == 0
            align_model, align_optimizer = clone_model_and_optimizer(model, optimizer)
            restore_rng_state(rng_state)
            apply_qk_diagonal_transform(align_model, scale, args.tele_layer, rope, low, high)
            alignment = compute_qk_muon_alignment(align_model, align_optimizer, batch, criterion, args.tele_layer)
            results.append(
                {
                    "candidate_index": index,
                    "identity": identity,
                    "scale": scale.detach().cpu().tolist(),
                    "scale_min": float(scale.min()),
                    "scale_max": float(scale.max()),
                    "alignment": float(alignment),
                    "alignment_rank": None,
                    "virtual_loss_evaluated": False,
                    "post_step_loss": None,
                    "searched": False,
                }
            )

        ranked = sorted(results, key=lambda item: float(item["alignment"]), reverse=True)
        for rank, item in enumerate(ranked, start=1):
            item["alignment_rank"] = rank
        shortlisted = [item for item in ranked if not item["identity"]][:shortlist_size]
        for item in shortlisted:
            shortlisted_indices.add(item["candidate_index"])

        identity_result = next(item for item in results if item["identity"])
        for check in [identity_result, *shortlisted]:
            if check["candidate_index"] in shortlisted_indices or check["identity"]:
                restore_rng_state(rng_state)
                step_model, step_optimizer = clone_model_and_optimizer(model, optimizer)
                restore_rng_state(rng_state)
                apply_qk_diagonal_transform(step_model, torch.tensor(check["scale"], dtype=torch.float32), args.tele_layer, rope, low, high)
                reset_qk_optimizer_state(step_model, step_optimizer, args.tele_layer)
                check["post_step_loss"] = float(virtual_optimizer_step(step_model, step_optimizer, batch, criterion))
                check["virtual_loss_evaluated"] = True
                check["searched"] = True

        restore_rng_state(rng_state)
        assert_live_state_unchanged(model, optimizer, live_snapshot)

        non_identity_results = [item for item in results if not item["identity"] and item["virtual_loss_evaluated"]]
        best_non_identity = min(non_identity_results, key=lambda item: float(item["post_step_loss"])) if non_identity_results else None
        epsilon = float(args.tele_accept_tol)

        identity_reset_loss = identity_result["post_step_loss"]
        beats_regular = (
            best_non_identity is not None
            and best_non_identity["post_step_loss"] < regular_loss - epsilon
        )
        beats_reset_only = (
            best_non_identity is not None
            and identity_reset_loss is not None
            and best_non_identity["post_step_loss"]
                < identity_reset_loss - epsilon
        )

        accepted = False
        reason = "rejected"
        if best_non_identity is not None and best_non_identity["post_step_loss"] < regular_loss - epsilon:
            accepted = True
            reason = "accepted_better_than_regular_muon"
        elif identity_result["virtual_loss_evaluated"] and identity_result["post_step_loss"] < regular_loss - epsilon:
            reason = "reset_only"
        else:
            reason = "rejected_did_not_beat_regular_muon"

        if accepted and best_non_identity is not None:
            apply_qk_diagonal_transform(
                model,
                torch.tensor(best_non_identity["scale"], dtype=torch.float32),
                args.tele_layer,
                rope,
                low,
                high,
            )
            reset_qk_optimizer_state(model, optimizer, args.tele_layer)

        if accepted and best_non_identity is not None:
            selected = best_non_identity
        else:
            selected = best_non_identity if best_non_identity is not None else identity_result

        post_loss_gain = float(regular_loss) - float(selected["post_step_loss"]) if selected["post_step_loss"] is not None else 0.0
        alignment_gain = float(selected["alignment"]) - float(identity_result["alignment"]) if identity_result is not None else 0.0

        telemetry = {
            "method": "muon_alignment",
            "accepted": accepted,
            "reason": reason,
            "step": step_index,
            "search_time_seconds": round(time.perf_counter() - started, 6),
            "regular_muon_post_step_loss": float(regular_loss),
            "identity_reset_loss": float(identity_result["post_step_loss"]),
            "identity_alignment": float(identity_result["alignment"]),
            "best_non_identity_candidate_loss": float(best_non_identity["post_step_loss"]) if best_non_identity is not None else None,
            "beats_regular_baseline": beats_regular,
            "beats_reset_only": beats_reset_only,
            "accepted_beyond_reset_only": accepted and beats_reset_only,
            "selected_alignment": float(selected["alignment"]),
            "selected_post_step_loss": float(selected["post_step_loss"]) if selected["post_step_loss"] is not None else None,
            "selected_identity": bool(selected["identity"]),
            "selected_scale": selected["scale"],
            "scale_min": selected["scale_min"],
            "scale_max": selected["scale_max"],
            "identity_rank": int(identity_result["alignment_rank"]),
            "state_reset": accepted,
            "tele_layer": args.tele_layer,
            "alignment_gain": alignment_gain,
            "post_loss_gain": post_loss_gain,
            "proposed_candidate_index": (best_non_identity["candidate_index"] if best_non_identity is not None else None),
            "proposed_post_loss_gain": (regular_loss - best_non_identity["post_step_loss"] if best_non_identity is not None and best_non_identity["post_step_loss"] is not None else None),
            "accepted_candidate_index": (best_non_identity["candidate_index"] if accepted and best_non_identity is not None else None),
            "accepted_post_loss_gain": (regular_loss - best_non_identity["post_step_loss"] if accepted and best_non_identity is not None else None),
            "shortlist_size": shortlist_size,
            "results": results,
        }
        return accepted, telemetry
    finally:
        restore_rng_state(rng_state)
        model.train(was_training)
        if not accepted:
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
