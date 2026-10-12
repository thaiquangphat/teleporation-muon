import math

import pytest
import torch
from torch import nn

from src.model import SimpleMLP
from src.muon import create_muon_optimizer
from src.teleport import (
    apply_diagonal_transform,
    apply_optimizer_state_mode,
    candidate_scales,
    identity_scale,
    try_teleportation,
    validate_bounds,
)


@pytest.fixture
def model():
    return SimpleMLP(input_dim=8, hidden_dim=4, num_classes=3)


def test_identity_transform_preserves_logits(model):
    x = torch.randn(5, 8)
    model.eval()
    base_logits = model(x)
    identity = identity_scale(model.hidden.out_features)
    apply_diagonal_transform(model, identity)
    assert torch.allclose(base_logits, model(x), atol=1e-6, rtol=1e-6)


def test_candidate_scales_are_positive_and_bounded():
    scales = candidate_scales(hidden_dim=6, low=0.7, high=1.5, count=5, seed=123)
    for scale in scales:
        assert torch.all(scale > 0)
        assert torch.all(scale >= 0.7)
        assert torch.all(scale <= 1.5)


def test_no_virtual_step_mutation(model):
    x = torch.randn(4, 8)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_muon_optimizer(model, lr=1e-2, weight_decay=1e-4)
    real_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    real_optimizer_state = {id(parameter): {key: value.clone() for key, value in optimizer.state.get(parameter, {}).items() if isinstance(value, torch.Tensor)} for parameter in optimizer.state}

    accepted, info = try_teleportation(model, optimizer, (x, torch.tensor([0, 1, 2, 0])), criterion, type("Args", (), {"tele_method": "one_step_loss", "tele_state_mode": "keep", "tele_candidates": 4, "tele_low": 0.7, "tele_high": 1.5, "tele_accept_tol": 1e-6, "seed": 7})(), 0)
    assert accepted in {True, False}
    for name, tensor in real_state.items():
        assert torch.allclose(model.state_dict()[name], tensor, atol=1e-7)


def test_bounds_reject_invalid_values():
    with pytest.raises(ValueError):
        validate_bounds(0.0, 2.0)
    with pytest.raises(ValueError):
        validate_bounds(2.0, 1.0)


def test_transport_scales_only_gradient_like_optimizer_state(model):
    optimizer = create_muon_optimizer(model, lr=1e-2, weight_decay=1e-4)
    parameters = dict(model.named_parameters())
    for parameter in parameters.values():
        optimizer.state[parameter] = {}
    optimizer.state[parameters["hidden.weight"]]["momentum_buffer"] = torch.ones_like(parameters["hidden.weight"])
    optimizer.state[parameters["hidden.bias"]]["exp_avg"] = torch.ones_like(parameters["hidden.bias"])
    optimizer.state[parameters["hidden.bias"]]["exp_avg_sq"] = torch.ones_like(parameters["hidden.bias"])
    optimizer.state[parameters["output.weight"]]["momentum_buffer"] = torch.ones_like(parameters["output.weight"])
    optimizer.state[parameters["output.bias"]]["exp_avg"] = torch.ones_like(parameters["output.bias"])
    scale = torch.tensor([0.5, 0.75, 1.25, 1.5])

    apply_optimizer_state_mode(model, optimizer, scale, "transport")

    assert torch.allclose(
        optimizer.state[parameters["hidden.weight"]]["momentum_buffer"],
        scale.reciprocal().unsqueeze(1).expand_as(parameters["hidden.weight"]),
    )
    assert torch.allclose(
        optimizer.state[parameters["hidden.bias"]]["exp_avg"],
        scale.reciprocal(),
    )
    assert torch.allclose(
        optimizer.state[parameters["hidden.bias"]]["exp_avg_sq"],
        scale.reciprocal().square(),
    )
    assert torch.equal(optimizer.state[parameters["output.bias"]]["exp_avg"], torch.ones_like(parameters["output.bias"]))


def test_training_step_runs_on_small_batch(model):
    x = torch.randn(8, 8)
    labels = torch.randint(0, 3, (8,))
    model.train()
    optimizer = create_muon_optimizer(model, lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
