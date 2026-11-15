"""GPT-2 teleportation, scoring, candidate selection, and state transport.

Candidate generation, candidate evaluation, score-based ranking, and winning
transformation commitment live in this module, matching the ViT structure.
Individual criterion modules under ``criteria/`` provide score formulas only.

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
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

try:  # package import
    from .muon import SingleDeviceMuonWithAuxAdam
    from .criteria import get_scoring_criterion
except ImportError:  # ``python src/lm/train.py`` import path
    from muon import SingleDeviceMuonWithAuxAdam
    from criteria import get_scoring_criterion


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
    """Generate the same per-head monomial APE transform used by ``src/vit``.

    Positive coordinate magnitudes are sampled from [low, high], followed by
    independent row/column permutations. The same per-head matrix is repeated
    across heads, matching the current ViT candidate generator.
    """
    low, high = validate_bounds(tele_low, tele_high)
    diagonal = torch.rand(d_head, dtype=torch.float32) * (high - low) + low
    if int(tele_sign) == 1:
        signs = torch.randint(0, 2, (d_head,), dtype=torch.int64).float() * 2.0 - 1.0
        diagonal = diagonal * signs
    base = torch.diag(diagonal)
    base = base[torch.randperm(d_head), :]
    matrix = base[:, torch.randperm(d_head)].contiguous()

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
    """Generate per-head scaled 2x2 rotations commuting with adjacent-pair RoPE."""
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
            c, sn = torch.cos(theta), torch.sin(theta)
            blocks.append(torch.stack((torch.stack((rho * c, -rho * sn)),
                                       torch.stack((rho * sn, rho * c)))))
            inv_rho = 1.0 / rho
            inverse_blocks.append(torch.stack((torch.stack((inv_rho * c, inv_rho * sn)),
                                                torch.stack((-inv_rho * sn, inv_rho * c)))))
        matrices.append(torch.block_diag(*blocks))
        inverses.append(torch.block_diag(*inverse_blocks))
    return _block_diag(matrices), _block_diag(inverses)


def gen_mlp_permutation(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Permutation symmetry, valid for any elementwise activation."""
    permutation = torch.eye(dim, dtype=torch.float32)[:, torch.randperm(dim)]
    return permutation, permutation.mT.contiguous()


def gen_mlp_inv_matrix(
    dim: int,
    tele_high: float,
    tele_low: float,
    activation_function: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use positive monomial transforms for ReLU, permutations otherwise.

    Positive rescaling commutes with ReLU by positive homogeneity. For GELU
    and the other non-homogeneous activations, only hidden-unit permutations
    are used as exact symmetries.
    """
    if str(activation_function).lower() == "relu":
        return gen_ape_inv_matrix(1, dim, tele_high, tele_low, tele_sign=0)
    return gen_mlp_permutation(dim)


def attention_blocks(model: nn.Module, tele_layer: str) -> List[nn.Module]:
    layers = list(model.transformer.h)
    if not layers:
        raise ValueError("The LM model has no transformer blocks")
    if tele_layer == "all":
        return layers
    if tele_layer == "first":
        return [layers[0]]
    if tele_layer == "last":
        return [layers[-1]]
    raise ValueError(f"Unsupported tele_layer: {tele_layer}")


def qk_linear_modules(model: nn.Module, tele_layer: str) -> List[Tuple[nn.Linear, nn.Linear]]:
    return [
        (block.attn.query, block.attn.key)
        for block in attention_blocks(model, tele_layer)
    ]


def _matrix_scale_summary(matrix: Optional[torch.Tensor]) -> List[float]:
    if matrix is None:
        return []
    return torch.linalg.vector_norm(matrix.float(), ord=2, dim=1).detach().cpu().tolist()


def generate_symmetry_candidates(
    model: nn.Module,
    args: Any,
    low: float,
    high: float,
    count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate identity plus candidates using the same conventions as ViT."""
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
                    matrix, inverse = gen_mlp_inv_matrix(
                        mlp_dim,
                        high,
                        low,
                        config.activation_function,
                    )
                candidate["M_mlp"], candidate["M_mlp_inv"] = matrix, inverse

            candidate["attention_scale_summary"] = _matrix_scale_summary(candidate["M_att"])
            candidate["mlp_scale_summary"] = _matrix_scale_summary(candidate["M_mlp"])
            candidate["scale_summary"] = (
                candidate["attention_scale_summary"] if use_attention
                else candidate["mlp_scale_summary"]
            )
            all_scales = candidate["attention_scale_summary"] + candidate["mlp_scale_summary"]
            candidate["scale_min"] = min(all_scales) if all_scales else 1.0
            candidate["scale_max"] = max(all_scales) if all_scales else 1.0
            candidate["transform_type"] = "gpt2_rope_block" if use_rope else "gpt2_ape_monomial"
            candidates.append(candidate)
    finally:
        restore_rng_state(original_rng)

    if not candidates or not candidates[0]["identity"]:
        raise ValueError("At least one candidate (identity) is required")
    return candidates


def teleportation_att(
    model: nn.Module,
    matrix: torch.Tensor,
    matrix_inv: torch.Tensor,
    tele_layer: str,
) -> None:
    """Apply the ViT-style Q/K symmetry to separate causal-LM projections.

    V and the attention output projection intentionally remain unchanged,
    matching the current ViT implementation and the author's reference code.
    """
    for query, key in qk_linear_modules(model, tele_layer):
        transform = matrix.to(device=query.weight.device, dtype=query.weight.dtype)
        inverse = matrix_inv.to(device=key.weight.device, dtype=key.weight.dtype)
        with torch.no_grad():
            query.weight.copy_(transform @ query.weight)
            key.weight.copy_(inverse.mT @ key.weight)
            if query.bias is not None:
                query.bias.copy_(transform @ query.bias)
            if key.bias is not None:
                key.bias.copy_(inverse.mT @ key.bias)


def teleportation_mlp(model: nn.Module, matrix: torch.Tensor, matrix_inv: torch.Tensor) -> None:
    """Apply an activation-compatible hidden-unit symmetry to every MLP block."""
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
            # proj.bias is in the residual stream and is unchanged.


def apply_symmetry_candidate(model: nn.Module, candidate: Dict[str, Any], args: Any) -> None:
    if bool(getattr(args, "tele_att", 1)):
        teleportation_att(model, candidate["M_att"], candidate["M_att_inv"], args.tele_layer)
    if bool(getattr(args, "tele_mlp", 0)):
        teleportation_mlp(model, candidate["M_mlp"], candidate["M_mlp_inv"])


def teleportation_parameter_items(
    model: nn.Module, args: Any
) -> List[Tuple[str, nn.Parameter]]:
    """List all parameters whose coordinates are changed by the active symmetry."""
    items: List[Tuple[str, nn.Parameter]] = []
    if bool(getattr(args, "tele_att", 1)):
        for name, parameter in model.named_parameters():
            for layer_index, _block in enumerate(model.transformer.h):
                if args.tele_layer == "first" and layer_index != 0:
                    continue
                if args.tele_layer == "last" and layer_index != len(model.transformer.h) - 1:
                    continue
                prefix = f"transformer.h.{layer_index}.attn."
                if name in {
                    prefix + "query.weight", prefix + "query.bias",
                    prefix + "key.weight", prefix + "key.bias",
                }:
                    items.append((name, parameter))
    if bool(getattr(args, "tele_mlp", 0)):
        for layer_index, _block in enumerate(model.transformer.h):
            prefix = f"transformer.h.{layer_index}.mlp."
            for suffix in ("c_fc.weight", "c_fc.bias", "c_proj.weight"):
                name = prefix + suffix
                parameter = dict(model.named_parameters()).get(name)
                if parameter is not None:
                    items.append((name, parameter))

    unique: List[Tuple[str, nn.Parameter]] = []
    seen = set()
    for name, parameter in items:
        if id(parameter) not in seen:
            unique.append((name, parameter))
            seen.add(id(parameter))
    return unique


def teleportation_matrix_parameter_names(model: nn.Module, args: Any) -> set[str]:
    """Names of transformed matrix weights included in CSD candidate scoring."""
    return {
        name for name, parameter in teleportation_parameter_items(model, args)
        if parameter.ndim >= 2 and name.endswith(".weight")
    }


def _transform_momentum_left(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    transform: torch.Tensor,
) -> None:
    state = optimizer.state.get(parameter)
    if not state:
        return
    momentum = state.get("momentum_buffer")
    if torch.is_tensor(momentum):
        matrix = transform.to(device=momentum.device, dtype=momentum.dtype)
        with torch.no_grad():
            momentum.copy_(matrix @ momentum)


def _transform_momentum_right(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    transform: torch.Tensor,
) -> None:
    state = optimizer.state.get(parameter)
    if not state:
        return
    momentum = state.get("momentum_buffer")
    if torch.is_tensor(momentum):
        matrix = transform.to(device=momentum.device, dtype=momentum.dtype)
        with torch.no_grad():
            momentum.copy_(momentum @ matrix)


def create_lm_muon_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
    adam_lr: float = 5e-5,
    adam_weight_decay: Optional[float] = None,
    nesterov: bool = False,
) -> torch.optim.Optimizer:
    """Use Muon for Transformer block matrices and AdamW for embeddings/biases/norms.

    This mirrors ``src/vit/muon.py``: hidden Transformer matrices are Muon
    parameters, while token/position embeddings and one-dimensional parameters
    use the auxiliary AdamW group.
    """
    muon_parameters: List[nn.Parameter] = []
    adam_parameters: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        is_hidden_matrix = parameter.ndim >= 2 and name.startswith("transformer.h.")
        (muon_parameters if is_hidden_matrix else adam_parameters).append(parameter)

    if not muon_parameters:
        raise ValueError("No Transformer block matrices were assigned to Muon")
    if not adam_parameters:
        raise ValueError("No embedding/bias/normalization parameters were assigned to AdamW")

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
    optimizer.parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    return optimizer


def transport_teleportation_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    candidate: Dict[str, Any],
    args: Any,
) -> None:
    """Transport Muon EMA buffers and reset only affected Adam states.

    For Q'=M Q and K'=M^{-T}K, gradient-like momentum transforms as
    B_Q'=M^{-T}B_Q and B_K'=M B_K. MLP state follows the same covector rule.
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
        for block in model.transformer.h:
            fc = block.mlp.c_fc
            proj = block.mlp.c_proj
            _transform_momentum_left(optimizer, fc.weight, matrix_inv.mT)
            _transform_momentum_right(optimizer, proj.weight, matrix.mT)
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


def _try_scored_teleportation_impl(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    args: Any,
    step_index: int,
    *,
    method_name: str,
    metrics_fn: Callable[[torch.optim.Optimizer], Dict[str, Any]],
    score_definition: str,
    requires_nesterov: bool,
) -> Tuple[bool, Dict[str, Any]]:
    """Select and commit the candidate maximizing one registered CSD score.

    Identity is always candidate zero and wins ties. Candidate models and
    optimizers are evaluated as clones; the live state is modified only after
    all candidates have been scored and the winner has been selected.
    """
    started = time.perf_counter()
    rng_state = capture_rng_state()
    live_snapshot = snapshot_live_state(model, optimizer)
    low, high = validate_bounds(args.tele_low, args.tele_high)
    candidate_count = int(args.tele_candidates)
    if candidate_count < 2:
        raise ValueError(
            f"{method_name} requires identity plus an alternative; "
            "set --tele-candidates >= 2"
        )

    try:
        from .criteria.common import muon_groups
    except ImportError:  # ``python src/lm/train.py`` import path
        from criteria.common import muon_groups
    groups = muon_groups(optimizer)
    if requires_nesterov and any(not bool(group.get("nesterov", False)) for group in groups):
        raise ValueError(
            f"{method_name} requires every Muon parameter group to have nesterov=True; "
            "construct the optimizer with --nesterov."
        )

    score_scope = str(getattr(args, "csd_score_scope", "all_muon"))
    if score_scope == "all_muon":
        # Exact parity with src/vit criteria: score every matrix in the Muon group.
        scoring_parameter_names = None
    elif score_scope == "transformed":
        scoring_parameter_names = teleportation_matrix_parameter_names(model, args)
        if not scoring_parameter_names:
            raise ValueError("The active teleportation configuration has no transformed Muon matrices to score")
    else:
        raise ValueError("csd_score_scope must be 'all_muon' or 'transformed'")

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
            seed=int(args.seed) + int(step_index),
        )
        if len(candidates) != candidate_count or not bool(candidates[0].get("identity", False)):
            raise RuntimeError("The LM candidate generator must return identity at index zero")

        results = []
        for index, candidate in enumerate(candidates):
            candidate_model, candidate_optimizer = clone_model_and_optimizer(model, optimizer)
            restore_rng_state(rng_state)
            if not bool(candidate["identity"]):
                apply_symmetry_candidate(candidate_model, candidate, args)
                transport_teleportation_optimizer_state(
                    candidate_model, candidate_optimizer, candidate, args
                )

            candidate_optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_batch(candidate_model, batch, criterion, args)
            loss.backward()
            metrics = metrics_fn(candidate_optimizer, parameter_names=scoring_parameter_names)
            score = float(metrics["score"])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite {method_name} score for candidate {index}")

            prefix = method_name
            result = {
                "candidate_index": index,
                "identity": bool(candidate["identity"]),
                "transform_type": candidate["transform_type"],
                "scale": candidate["scale_summary"],
                "scale_min": float(candidate["scale_min"]),
                "scale_max": float(candidate["scale_max"]),
                "attention_scale": candidate["attention_scale_summary"],
                "mlp_scale": candidate["mlp_scale_summary"],
                "selection_score": score,
                "score": score,
                "squared_score": float(metrics["squared_score"]),
                "matrix_count": int(metrics["matrix_count"]),
                "matrix_norms": metrics["matrix_norms"],
                "matrix_names": metrics["matrix_names"],
                "max_matrix_norm": float(metrics["max_matrix_norm"]),
                "sum_matrix_norms": float(metrics["sum_matrix_norms"]),
                "norm_kind": str(metrics["norm_kind"]),
                "momentum_direction": str(metrics["momentum_direction"]),
                "beta": float(metrics["beta"]),
                "scoring_parameter_scope": (
                    "all_muon_matrices" if score_scope == "all_muon" else "transformed_muon_matrices"
                ),
                f"{prefix}_score": score,
                f"{prefix}_squared_score": float(metrics["squared_score"]),
                f"{prefix}_matrix_count": int(metrics["matrix_count"]),
                f"{prefix}_matrix_norms": metrics["matrix_norms"],
                f"{prefix}_matrix_names": metrics["matrix_names"],
                f"{prefix}_norm_kind": str(metrics["norm_kind"]),
                f"{prefix}_momentum_direction": str(metrics["momentum_direction"]),
            }

            # Keep the telemetry keys emitted by the previous LM CSD-L2
            # Nesterov implementation for existing analysis notebooks.
            if method_name == "csd_l2_nesterov":
                result.update({
                    "csd_l2_score": score,
                    "csd_l2_squared_score": float(metrics["squared_score"]),
                    "csd_l2_matrix_count": int(metrics["matrix_count"]),
                    "csd_l2_max_matrix_nuclear_norm": float(metrics["max_matrix_norm"]),
                    "csd_l2_sum_matrix_nuclear_norms": float(metrics["sum_matrix_norms"]),
                    "csd_l2_matrix_nuclear_norms": metrics["matrix_norms"],
                    "csd_l2_matrix_names": metrics["matrix_names"],
                    "csd_l2_momentum_direction": str(metrics["momentum_direction"]),
                })
            results.append(result)
            del candidate_model, candidate_optimizer, loss, metrics

        ranked = sorted(
            results,
            key=lambda item: (-float(item["selection_score"]), int(item["candidate_index"])),
        )
        for rank, item in enumerate(ranked, start=1):
            item[f"{method_name}_rank"] = rank
            if method_name == "csd_l2_nesterov":
                item["csd_l2_rank"] = rank

        selected = ranked[0]
        selected_index = int(selected["candidate_index"])
        selected_candidate = candidates[selected_index]
        identity_result = next(item for item in results if item["identity"])
        best_alternative = max(
            (item for item in results if not item["identity"]),
            key=lambda item: float(item["selection_score"]),
        )

        restore_rng_state(rng_state)
        model.train(was_training)
        assert_live_state_unchanged(model, optimizer, live_snapshot)

        accepted = not bool(selected["identity"])
        if accepted:
            apply_symmetry_candidate(model, selected_candidate, args)
            transport_teleportation_optimizer_state(model, optimizer, selected_candidate, args)
            committed = True
            reason = f"selected_max_{method_name}_score"
        else:
            reason = f"identity_selected_max_{method_name}_score_noop"

        identity_score = float(identity_result["selection_score"])
        selected_score = float(selected["selection_score"])
        gain = selected_score - identity_score
        telemetry = {
            "method": method_name,
            "selection_metric": method_name,
            "baseline_score": identity_score,
            "selected_score": selected_score,
            "score_gain": gain,
            "accepted": bool(accepted),
            "reason": reason,
            "step": int(step_index),
            "search_time_seconds": round(time.perf_counter() - started, 6),
            "identity_score": identity_score,
            "selected_metric_score": selected_score,
            "metric_gain": gain,
            "selected_candidate_index": selected_index,
            "selected_identity": bool(selected["identity"]),
            "selected_scale": selected["scale"],
            "scale_min": float(selected["scale_min"]),
            "scale_max": float(selected["scale_max"]),
            "selected_attention_scale": selected_candidate["attention_scale_summary"],
            "selected_mlp_scale": selected_candidate["mlp_scale_summary"],
            "state_reset": False,
            "muon_momentum_reset": False,
            "auxiliary_adam_state_reset": bool(accepted),
            "momentum_state_transported": bool(accepted),
            "metric_definition": score_definition,
            "selection_objective": (
                f"euclidean_aggregate_of_matrix_{selected['norm_kind']}_norms"
            ),
            "momentum_direction": selected["momentum_direction"],
            "matrix_count": int(selected["matrix_count"]),
            "matrix_norms": selected["matrix_norms"],
            "matrix_names": selected["matrix_names"],
            "norm_kind": selected["norm_kind"],
            "beta": float(selected["beta"]),
            "tele_layer": args.tele_layer,
            "transform_family": selected_candidate["transform_type"],
            "uses_attention": bool(getattr(args, "tele_att", 1)),
            "uses_mlp": bool(getattr(args, "tele_mlp", 0)),
            "scoring_parameter_scope": (
                "all_muon_matrices" if score_scope == "all_muon" else "transformed_muon_matrices"
            ),
            "proposed_candidate_index": int(best_alternative["candidate_index"]),
            "accepted_candidate_index": selected_index if accepted else None,
            "num_candidates_evaluated": len(results),
            "results": results,
        }

        if method_name == "csd_l2_nesterov":
            # Preserve the exact legacy fields used in previous LM runs.
            telemetry.update({
                "selection_metric": "nesterov_csd_l2",
                "identity_csd_l2_score": identity_score,
                "selected_csd_l2_score": selected_score,
                "csd_l2_gain": gain,
                "csd_l2_definition": score_definition,
                "csd_l2_matrix_count": int(selected["matrix_count"]),
                "csd_l2_matrix_nuclear_norms": selected["matrix_norms"],
                "csd_l2_matrix_names": selected["matrix_names"],
                "momentum_direction": "nesterov_corrected_pre_newton_schulz",
                "selection_objective": (
                    "euclidean_aggregate_of_matrix_nuclear_norms_of_nesterov_direction"
                ),
            })
        return accepted, telemetry
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
    return _try_scored_teleportation_impl(
        model, optimizer, batch, criterion, args, step_index,
        metrics_fn=criterion_spec["metrics_fn"],
        method_name=criterion_spec["method_name"],
        score_definition=criterion_spec["score_definition"],
        requires_nesterov=bool(criterion_spec.get("requires_nesterov", False)),
    )


def try_csd_l2_teleportation(model, optimizer, batch, criterion, args, step_index):
    return try_scored_teleportation(
        model, optimizer, batch, criterion, args, step_index, "csd_l2"
    )


def try_csd_l2_nesterov_teleportation(model, optimizer, batch, criterion, args, step_index):
    return try_scored_teleportation(
        model, optimizer, batch, criterion, args, step_index, "csd_l2_nesterov"
    )


def try_csd_l2_nesterov_frobenius_teleportation(
    model, optimizer, batch, criterion, args, step_index
):
    return try_scored_teleportation(
        model, optimizer, batch, criterion, args, step_index,
        "csd_l2_nesterov_frobenius"
    )


def try_csd_l2_nesterov_post_orth_teleportation(
    model, optimizer, batch, criterion, args, step_index
):
    return try_scored_teleportation(
        model, optimizer, batch, criterion, args, step_index,
        "csd_l2_nesterov_post_orth"
    )


def try_csd_l2_nesterov_post_orth_frobenius_teleportation(
    model, optimizer, batch, criterion, args, step_index
):
    return try_scored_teleportation(
        model, optimizer, batch, criterion, args, step_index,
        "csd_l2_nesterov_post_orth_frobenius"
    )


def try_muon_alignment_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Backward-compatible alias for the registered Nesterov CSD criterion."""
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
    """Dispatch teleportation; candidate search and commit stay in this module."""
    method = getattr(args, "tele_method", "none")
    if method in {"none", None}:
        return False, {"method": "none", "accepted": False, "step": int(step_index)}
    if method == "naive":
        try:
            from .criteria.naive import try_naive_teleportation
        except ImportError:  # ``python src/lm/train.py`` import path
            from criteria.naive import try_naive_teleportation
        return try_naive_teleportation(
            model, optimizer, batch, criterion, args, step_index
        )

    # Preserve the previous LM alias: muon_alignment means Nesterov CSD-L2.
    if method == "muon_alignment":
        method = "csd_l2_nesterov"
    supported = {
        "csd_l2",
        "csd_l2_nesterov",
        "csd_l2_nesterov_frobenius",
        "csd_l2_nesterov_post_orth",
        "csd_l2_nesterov_post_orth_frobenius",
    }
    if method not in supported:
        raise ValueError(
            f"Unsupported LM teleportation method {method!r}; "
            f"supported methods: none, naive, {', '.join(sorted(supported))}, "
            "muon_alignment."
        )
    return try_scored_teleportation(
        model, optimizer, batch, criterion, args, step_index, method
    )


__all__ = [
    "create_lm_muon_optimizer",
    "generate_tele_scheduler",
    "generate_symmetry_candidates",
    "teleportation_parameter_items",
    "teleportation_matrix_parameter_names",
    "teleportation_att",
    "teleportation_mlp",
    "transport_teleportation_optimizer_state",
    "try_scored_teleportation",
    "try_csd_l2_teleportation",
    "try_csd_l2_nesterov_teleportation",
    "try_csd_l2_nesterov_frobenius_teleportation",
    "try_csd_l2_nesterov_post_orth_teleportation",
    "try_csd_l2_nesterov_post_orth_frobenius_teleportation",
    "try_muon_alignment_teleportation",
    "try_teleportation",
]
