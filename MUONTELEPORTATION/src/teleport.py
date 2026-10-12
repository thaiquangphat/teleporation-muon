import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from muon import create_muon_optimizer

@dataclass
class CandidateResult:
    scale: torch.Tensor
    score: float
    loss_before: float
    loss_after: float | None = None
    accepted: bool = False
    method: str = ""


def validate_bounds(low: float, high: float) -> Tuple[float, float]:
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ValueError(f"Teleport bounds must satisfy 0 < low < high; got low={low}, high={high}.")
    return low, high


def identity_scale(hidden_dim: int) -> torch.Tensor:
    return torch.ones(hidden_dim, dtype=torch.float32)


def candidate_scales(hidden_dim: int, low: float, high: float, count: int, seed: int) -> List[torch.Tensor]:
    validate_bounds(low, high)
    rng = random.Random(seed)
    candidates: List[torch.Tensor] = [identity_scale(hidden_dim)]
    log_low = math.log(low)
    log_high = math.log(high)
    for _ in range(max(0, count - 1)):
        for _ in range(300):
            log_scales = torch.empty(hidden_dim, dtype=torch.float32).uniform_(log_low, log_high)
            log_scales = log_scales - log_scales.mean()
            scales = torch.exp(log_scales)
            if torch.all(scales > 0) and torch.all(scales >= low) and torch.all(scales <= high):
                candidates.append(scales)
                break
        else:
            log_scales = torch.linspace(log_low, log_high, steps=hidden_dim)
            log_scales = log_scales - log_scales.mean()
            scales = torch.exp(log_scales)
            candidate = scales / scales.mean().clamp_min(1e-8)
            candidates.append(candidate)
    return candidates


def apply_diagonal_transform(model: nn.Module, diag: torch.Tensor) -> None:
    if not hasattr(model, "hidden") or not hasattr(model, "output"):
        raise TypeError("apply_diagonal_transform expects a two-layer ReLU MLP with hidden and output attributes.")

    hidden = model.hidden
    output = model.output
    diag = diag.to(device=hidden.weight.device, dtype=hidden.weight.dtype)
    inv_diag = torch.reciprocal(diag).to(device=output.weight.device, dtype=output.weight.dtype)

    with torch.no_grad():
        hidden.weight.mul_(diag.unsqueeze(1))
        if hidden.bias is not None:
            hidden.bias.mul_(diag)
        output.weight.mul_(inv_diag.unsqueeze(0))
        if output.bias is not None:
            output.bias.mul_(1.0)


def restore_diagonal_transform(model: nn.Module, diag: torch.Tensor) -> None:
    if not hasattr(model, "hidden") or not hasattr(model, "output"):
        raise TypeError("restore_diagonal_transform expects a two-layer ReLU MLP with hidden and output attributes.")

    hidden = model.hidden
    output = model.output
    diag = diag.to(device=hidden.weight.device, dtype=hidden.weight.dtype)
    inv_diag = torch.reciprocal(diag).to(device=output.weight.device, dtype=output.weight.dtype)

    with torch.no_grad():
        hidden.weight.mul_(inv_diag.unsqueeze(1))
        if hidden.bias is not None:
            hidden.bias.mul_(inv_diag)
        output.weight.mul_(diag.unsqueeze(0))


def compute_grad_norm(model: nn.Module) -> float:
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not grads:
        return 0.0
    return sum(grad.norm().item() ** 2 for grad in grads) ** 0.5


def _make_optimizer_for_model(model: nn.Module, optimizer: torch.optim.Optimizer):
    if isinstance(optimizer, torch.optim.AdamW):
        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer.param_groups[0]["lr"],
            betas=optimizer.param_groups[0].get("betas", (0.9, 0.999)),
            eps=optimizer.param_groups[0].get("eps", 1e-8),
            weight_decay=optimizer.param_groups[0].get("weight_decay", 0.0),
        )
    if isinstance(optimizer, torch.optim.SGD):
        return torch.optim.SGD(
            model.parameters(),
            lr=optimizer.param_groups[0]["lr"],
            momentum=optimizer.param_groups[0].get("momentum", 0.0),
            weight_decay=optimizer.param_groups[0].get("weight_decay", 0.0),
        )
    if optimizer.__class__.__name__.endswith("MuonWithAuxAdam") or "SingleDeviceMuonWithAuxAdam" in optimizer.__class__.__name__:
        return create_muon_optimizer(
            model,
            lr=optimizer.param_groups[0]["lr"],
            weight_decay=optimizer.param_groups[0].get("weight_decay", 0.0),
            momentum=optimizer.param_groups[0].get("momentum", 0.95),
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=optimizer.param_groups[0]["lr"],
        weight_decay=optimizer.param_groups[0].get("weight_decay", 0.0),
    )


def clone_state_into_optimizer(model: nn.Module, model_clone: nn.Module, optimizer: torch.optim.Optimizer, optimizer_clone: torch.optim.Optimizer) -> None:
    real_named = {name: parameter for name, parameter in model.named_parameters()}
    clone_named = {name: parameter for name, parameter in model_clone.named_parameters()}
    for name, parameter in real_named.items():
        if name not in clone_named:
            continue
        state = optimizer.state.get(parameter)
        if state is None:
            continue
        clone_state = optimizer_clone.state.get(clone_named[name])
        if clone_state is None:
            clone_state = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                clone_state[key] = value.detach().clone()
            else:
                clone_state[key] = copy.deepcopy(value)
        optimizer_clone.state[clone_named[name]] = clone_state


def apply_optimizer_state_mode(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    diag: torch.Tensor,
    state_mode: str,
) -> None:
    if state_mode == "keep":
        return
    if state_mode not in {"transport", "reset"}:
        raise ValueError(f"Unsupported teleport state mode: {state_mode}")

    diag = diag.to(device=model.hidden.weight.device, dtype=model.hidden.weight.dtype)
    inverse = diag.reciprocal()
    transformed = {
        "hidden.weight": (inverse.unsqueeze(1), inverse.square().unsqueeze(1)),
        "hidden.bias": (inverse, inverse.square()),
        "output.weight": (diag.unsqueeze(0), diag.square().unsqueeze(0)),
    }
    parameters = dict(model.named_parameters())

    for name, (linear_scale, squared_scale) in transformed.items():
        parameter = parameters[name]
        if state_mode == "reset":
            optimizer.state.pop(parameter, None)
            continue

        state = optimizer.state.get(parameter)
        if not state:
            continue
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key in {"momentum_buffer", "exp_avg"}:
                value.mul_(linear_scale.to(device=value.device, dtype=value.dtype))
            elif key in {"exp_avg_sq", "max_exp_avg_sq"}:
                value.mul_(squared_scale.to(device=value.device, dtype=value.dtype))


def score_candidate(model: nn.Module, optimizer: torch.optim.Optimizer, batch: Tuple[torch.Tensor, torch.Tensor], criterion: nn.Module, method: str, candidate: torch.Tensor, state_mode: str, low: float, high: float, accept_tol: float, seed: int) -> Dict[str, Any]:
    model_clone = copy.deepcopy(model)
    optimizer_clone = _make_optimizer_for_model(model_clone, optimizer)
    clone_state_into_optimizer(model, model_clone, optimizer, optimizer_clone)

    apply_diagonal_transform(model_clone, candidate)
    apply_optimizer_state_mode(model_clone, optimizer_clone, candidate, state_mode)
    logits = model_clone(batch[0])
    loss = criterion(logits, batch[1])
    loss.backward()

    if method == "grad_norm":
        return {"score": compute_grad_norm(model_clone), "loss_after": float(loss.detach()), "identity": False}
    if method == "muon_alignment":
        dot_total = 0.0
        for parameter in model_clone.parameters():
            if parameter.grad is None:
                continue
            for state in optimizer_clone.state.values():
                if "momentum_buffer" in state and state["momentum_buffer"].shape == parameter.grad.shape:
                    dot_total += float((parameter.grad * state["momentum_buffer"]).sum().item())
        return {"score": dot_total, "loss_after": float(loss.detach()), "identity": False}
    if method == "one_step_loss":
        optimizer_clone.step()
        with torch.no_grad():
            post_loss = criterion(model_clone(batch[0]), batch[1]).item()
        return {"score": float(post_loss), "loss_after": float(post_loss), "identity": False}
    return {"score": float(loss.detach()), "loss_after": float(loss.detach()), "identity": False}


def apply_teleportation_to_real_model(model: nn.Module, diag: torch.Tensor, state_mode: str, optimizer: torch.optim.Optimizer) -> None:
    apply_diagonal_transform(model, diag)
    apply_optimizer_state_mode(model, optimizer, diag, state_mode)


def try_teleportation(model: nn.Module, optimizer: torch.optim.Optimizer, batch: Tuple[torch.Tensor, torch.Tensor], criterion: nn.Module, args: Any, step_index: int) -> Tuple[bool, Dict[str, Any]]:
    low, high = validate_bounds(args.tele_low, args.tele_high)
    hidden_dim = model.hidden.out_features
    candidate_pool = candidate_scales(hidden_dim, low, high, count=args.tele_candidates, seed=args.seed + step_index)
    base_loss = criterion(model(batch[0]), batch[1]).item()
    best: Dict[str, Any] = {"scale": identity_scale(hidden_dim), "score": float("inf"), "loss_after": base_loss, "identity": True}
    result_log: List[Dict[str, Any]] = []

    for idx, scale in enumerate(candidate_pool):
        if args.tele_method == "none":
            return False, {"method": "none", "accepted": False, "candidate": None, "step": step_index, "base_loss": base_loss}

        model_candidate = copy.deepcopy(model)
        optimizer_candidate = _make_optimizer_for_model(model_candidate, optimizer)
        clone_state_into_optimizer(model, model_candidate, optimizer, optimizer_candidate)
        apply_diagonal_transform(model_candidate, scale)
        apply_optimizer_state_mode(model_candidate, optimizer_candidate, scale, args.tele_state_mode)

        logits = model_candidate(batch[0])
        loss = criterion(logits, batch[1])
        loss.backward()
        if args.tele_method == "grad_norm":
            score = compute_grad_norm(model_candidate)
            better = score > (best["score"] if best["score"] != float("inf") else -1e30)
        elif args.tele_method == "muon_alignment":
            score = 0.0
            for parameter in model_candidate.parameters():
                if parameter.grad is None:
                    continue
                for state in optimizer_candidate.state.values():
                    if "momentum_buffer" in state and state["momentum_buffer"].shape == parameter.grad.shape:
                        score += float((parameter.grad * state["momentum_buffer"]).sum().item())
            better = score > (best["score"] if best["score"] != float("inf") else -1e30)
        elif args.tele_method == "one_step_loss":
            optimizer_candidate.step()
            score = criterion(model_candidate(batch[0]), batch[1]).item()
            better = score < best["score"]
        elif args.tele_method == "naive":
            score = float(loss.item())
            better = False
        else:
            score = float(loss.item())
            better = score < best["score"]

        if better:
            best = {
                "scale": scale.detach().clone(),
                "score": float(score),
                "loss_after": float(loss.item()),
                "identity": idx == 0,
                "idx": idx,
            }
        result_log.append({"candidate_index": idx, "scale": scale.tolist(), "score": float(score), "identity": idx == 0})

    identity_reference = next(item for item in result_log if item["identity"])
    selected = best
    selected_score = selected["score"]
    if args.tele_method == "naive" and len(candidate_pool) > 1:
        selected_index = random.Random(args.seed + step_index).randrange(1, len(candidate_pool))
        selected = {
            "scale": candidate_pool[selected_index],
            "score": result_log[selected_index]["score"],
            "identity": False,
        }
        selected_score = selected["score"]
        accepted = True
    elif args.tele_method in {"grad_norm", "muon_alignment"}:
        accepted = bool(selected_score > identity_reference["score"] + args.tele_accept_tol)
    elif args.tele_method == "one_step_loss":
        accepted = bool(selected_score < identity_reference["score"] - args.tele_accept_tol)
    else:
        accepted = False

    if accepted:
        return True, {
            "method": args.tele_method,
            "accepted": True,
            "candidate": selected["scale"].tolist(),
            "scale_min": float(selected["scale"].min()),
            "scale_max": float(selected["scale"].max()),
            "base_loss": base_loss,
            "step": step_index,
            "results": result_log,
        }

    return False, {
        "method": args.tele_method,
        "accepted": False,
        "candidate": None,
        "base_loss": base_loss,
        "step": step_index,
        "results": result_log,
    }
