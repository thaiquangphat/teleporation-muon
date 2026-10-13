import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import ViTConfig

from model import ViTRoPEForImageClassification
from muon import _muon_update, create_muon_optimizer
from teleport import (
    apply_qk_diagonal_transform,
    assert_live_state_unchanged,
    candidate_qk_scales,
    capture_rng_state,
    clone_model_and_optimizer,
    compute_qk_muon_alignment,
    identity_scale,
    qk_parameter_items,
    qk_weight_parameters,
    reset_qk_optimizer_state,
    restore_rng_state,
    snapshot_live_state,
    try_naive_teleportation,
    try_teleportation,
    validate_bounds,
    validate_qk_scale,
    virtual_optimizer_step,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def tiny_config(**overrides):
    config = {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "image_size": 8,
        "patch_size": 4,
        "num_channels": 1,
        "num_labels": 4,
        "qkv_bias": True,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
    }
    config.update(overrides)
    return ViTConfig(**config)


def tiny_model(**overrides):
    torch.manual_seed(0)
    return ViTRoPEForImageClassification(tiny_config(**overrides))


def tiny_batch(model, batch_size=4):
    channels = model.config.num_channels
    size = model.config.image_size
    images = torch.randn(batch_size, channels, size, size)
    labels = torch.randint(0, model.config.num_labels, (batch_size,))
    return images, labels


def teleport_args(**overrides):
    args = dict(
        tele_method="muon_alignment",
        tele_layer="all",
        tele_candidates=3,
        tele_low=0.5,
        tele_high=2.0,
        tele_accept_tol=1e-6,
        position_embedding="rope",
        seed=0,
        n_teleport=4,
        tele_att=1,
        tele_mlp=0,
        num_heads=2,
        d_model=16,
        tele_sign=0,
        intermediate_size=32,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def load_reference_model_class():
    path = REPO_ROOT / "MNISTCIFAR" / "src" / "model.py"
    spec = importlib.util.spec_from_file_location("mnistcifar_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ViTRoPEForImageClassification


def test_vit_matches_mnistcifar_reference_logits():
    config = tiny_config()
    torch.manual_seed(123)
    reference = load_reference_model_class()(config)
    torch.manual_seed(123)
    reimplemented = ViTRoPEForImageClassification(config)
    missing, unexpected = reimplemented.load_state_dict(reference.state_dict(), strict=True)
    assert missing == []
    assert unexpected == []

    images = torch.randn(3, config.num_channels, config.image_size, config.image_size)
    reference.eval()
    reimplemented.eval()
    with torch.no_grad():
        reference_logits = reference(pixel_values=images).logits
        reimplemented_logits = reimplemented(pixel_values=images).logits
    assert torch.allclose(reference_logits, reimplemented_logits, atol=1e-6, rtol=1e-6)


def test_qk_identity_and_tied_rope_scales_preserve_logits():
    model = tiny_model()
    model.eval()
    images, _ = tiny_batch(model)
    with torch.no_grad():
        base = model(pixel_values=images).logits

    apply_qk_diagonal_transform(model, identity_scale(model.config.hidden_size), "all", True, 0.5, 2.0)
    with torch.no_grad():
        identity_logits = model(pixel_values=images).logits
    assert torch.allclose(base, identity_logits, atol=1e-6, rtol=1e-6)

    scale = torch.tensor([1.3, 1.3, 0.7, 0.7] * (model.config.hidden_size // 4), dtype=torch.float32)
    apply_qk_diagonal_transform(model, scale, "all", True, 0.5, 2.0)
    with torch.no_grad():
        tied_logits = model(pixel_values=images).logits
    assert torch.allclose(base, tied_logits, atol=1e-5, rtol=1e-5)


def test_invalid_rope_pair_scales_are_rejected():
    scale = torch.tensor([1.2, 0.8, 1.1, 1.1])
    with pytest.raises(ValueError, match="tied within each adjacent rotary pair"):
        validate_qk_scale(scale, 0.5, 2.0, rope=True)
    with pytest.raises(ValueError):
        validate_bounds(0.0, 2.0)
    with pytest.raises(ValueError):
        apply_qk_diagonal_transform(tiny_model(), scale, "all", True, 0.5, 2.0)


def test_candidate_scales_are_positive_finite_bounded_and_include_identity():
    scales = candidate_qk_scales(dim=8, low=0.5, high=1.8, count=5, seed=3, rope=True)
    assert torch.equal(scales[0], identity_scale(8))
    for scale in scales:
        validate_qk_scale(scale, 0.5, 1.8, rope=True)


def test_alignment_uses_only_qk_weights_and_reset_muon_direction():
    model = tiny_model()
    images, labels = tiny_batch(model)
    optimizer = create_muon_optimizer(model, lr=0.02, weight_decay=1e-4, momentum=0.95)
    criterion = nn.CrossEntropyLoss()
    score = compute_qk_muon_alignment(model, optimizer, (images, labels), criterion, "all")

    expected = 0.0
    other = 0.0
    qk_ids = {id(parameter) for parameter in qk_weight_parameters(model, "all")}
    for group in optimizer.param_groups:
        if not group.get("use_muon"):
            continue
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            update = _muon_update(parameter.grad.detach().clone(), torch.zeros_like(parameter.grad), beta=group["momentum"])
            contrib = float(group["lr"]) * float(torch.sum(parameter.grad.detach().float().reshape_as(update) * update.float()).item())
            other += contrib
            if id(parameter) in qk_ids:
                expected += contrib
    assert score == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert other != pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_optimizer_clone_and_identity_virtual_step_do_not_touch_live_state():
    model = tiny_model()
    images, labels = tiny_batch(model)
    optimizer = create_muon_optimizer(model, lr=0.02, weight_decay=1e-4, momentum=0.95)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model(pixel_values=images).logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    rng_before = capture_rng_state()
    snapshot = snapshot_live_state(model, optimizer)
    clone_model, clone_optimizer = clone_model_and_optimizer(model, optimizer)
    named_live = dict(model.named_parameters())
    named_clone = dict(clone_model.named_parameters())
    for name, parameter in named_live.items():
        live_state = optimizer.state.get(parameter)
        clone_state = clone_optimizer.state.get(named_clone[name])
        if not live_state:
            continue
        for key, value in live_state.items():
            if torch.is_tensor(value):
                assert torch.equal(value, clone_state[key])
                clone_state[key].add_(1.0)
                assert not torch.equal(value, clone_state[key])

    reset_qk_optimizer_state(clone_model, clone_optimizer, "all")
    virtual_optimizer_step(clone_model, clone_optimizer, (images, labels), criterion)
    restore_rng_state(rng_before)
    assert_live_state_unchanged(model, optimizer, snapshot)


def test_virtual_search_leaves_live_rng_and_state_unchanged(monkeypatch):
    model = tiny_model()
    images, labels = tiny_batch(model)
    optimizer = create_muon_optimizer(model, lr=0.02, weight_decay=1e-4, momentum=0.95)
    criterion = nn.CrossEntropyLoss()
    snapshot = snapshot_live_state(model, optimizer)
    rng_before = capture_rng_state()

    def fake_virtual(model, optimizer, batch, criterion):
        torch.rand(2)
        random_state = torch.randint(0, 10, (4,))
        return float(random_state.float().mean())

    monkeypatch.setattr("teleport.virtual_optimizer_step", fake_virtual)
    monkeypatch.setattr("teleport.compute_qk_muon_alignment", lambda *args, **kwargs: 0.0)
    try_teleportation(model, optimizer, (images, labels), criterion, teleport_args(tele_candidates=2), 0)
    rng_after = capture_rng_state()
    assert torch.equal(rng_before["torch_cpu"], rng_after["torch_cpu"])
    assert rng_before["python"] == rng_after["python"]
    assert_live_state_unchanged(model, optimizer, snapshot)


def test_better_alignment_but_worse_loss_is_rejected(monkeypatch):
    model = tiny_model()
    images, labels = tiny_batch(model)
    optimizer = create_muon_optimizer(model, lr=0.02, weight_decay=1e-4, momentum=0.95)
    criterion = nn.CrossEntropyLoss()
    snapshot = snapshot_live_state(model, optimizer)

    alignments = iter([0.1, 9.0])
    losses = iter([1.00, 1.20, 1.10])

    monkeypatch.setattr("teleport.compute_qk_muon_alignment", lambda *args, **kwargs: next(alignments))
    monkeypatch.setattr("teleport.virtual_optimizer_step", lambda *args, **kwargs: next(losses))

    accepted, info = try_teleportation(
        model,
        optimizer,
        (images, labels),
        criterion,
        teleport_args(tele_candidates=2, tele_accept_tol=1e-4),
        0,
    )
    assert accepted is False
    assert info["reason"] == "rejected_did_not_beat_regular_muon"
    assert info["results"][1]["alignment"] > info["results"][0]["alignment"]
    assert info["results"][1]["post_step_loss"] > info["regular_muon_post_step_loss"]
    assert_live_state_unchanged(model, optimizer, snapshot)


def test_candidate_accepted_only_when_it_beats_regular_muon(monkeypatch):
    model = tiny_model()
    images, labels = tiny_batch(model)
    optimizer = create_muon_optimizer(model, lr=0.02, weight_decay=1e-4, momentum=0.95)
    criterion = nn.CrossEntropyLoss()

    alignments = iter([0.1, 0.2])
    losses = iter([1.00, 1.20, 0.80])
    monkeypatch.setattr("teleport.compute_qk_muon_alignment", lambda *args, **kwargs: next(alignments))
    monkeypatch.setattr("teleport.virtual_optimizer_step", lambda *args, **kwargs: next(losses))

    accepted, info = try_teleportation(
        model,
        optimizer,
        (images, labels),
        criterion,
        teleport_args(tele_candidates=2, tele_accept_tol=1e-4),
        0,
    )
    assert accepted is True
    assert info["reason"] == "accepted_better_than_regular_muon"
    assert info["selected_post_step_loss"] < info["regular_muon_post_step_loss"] - 1e-4
    assert info["state_reset"] is True


def test_identity_reset_winner_is_logged_as_reset_only(monkeypatch):
    model = tiny_model()
    images, labels = tiny_batch(model)
    optimizer = create_muon_optimizer(model, lr=0.02, weight_decay=1e-4, momentum=0.95)
    criterion = nn.CrossEntropyLoss()
    snapshot = snapshot_live_state(model, optimizer)

    alignments = iter([1.0, 2.0])
    losses = iter([1.50, 0.90, 1.00])
    monkeypatch.setattr("teleport.compute_qk_muon_alignment", lambda *args, **kwargs: next(alignments))
    monkeypatch.setattr("teleport.virtual_optimizer_step", lambda *args, **kwargs: next(losses))

    accepted, info = try_teleportation(
        model,
        optimizer,
        (images, labels),
        criterion,
        teleport_args(tele_candidates=2),
        0,
    )
    assert accepted is False
    assert info["reason"] == "reset_only"
    assert_live_state_unchanged(model, optimizer, snapshot)


def _fill_optimizer_state(model, optimizer, images, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model(pixel_values=images).logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_rejection_does_not_change_live_parameters_or_state(monkeypatch):
    model = tiny_model()
    images, labels = tiny_batch(model)
    optimizer = create_muon_optimizer(model, lr=0.02, weight_decay=1e-4, momentum=0.95)
    _fill_optimizer_state(model, optimizer, images, labels)
    snapshot = snapshot_live_state(model, optimizer)
    monkeypatch.setattr("teleport.compute_qk_muon_alignment", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr("teleport.virtual_optimizer_step", lambda *args, **kwargs: 1.0)
    accepted, info = try_teleportation(
        model,
        optimizer,
        (images, labels),
        nn.CrossEntropyLoss(),
        teleport_args(tele_candidates=2),
        0,
    )
    assert accepted is False
    assert info["reason"] == "reset_only"
    assert_live_state_unchanged(model, optimizer, snapshot)


def test_accepted_transform_resets_only_qk_state_before_one_real_step(monkeypatch):
    model = tiny_model()
    images, labels = tiny_batch(model)
    optimizer = create_muon_optimizer(model, lr=0.02, weight_decay=1e-4, momentum=0.95)
    _fill_optimizer_state(model, optimizer, images, labels)
    named = dict(model.named_parameters())
    qk_names = {name for name, _ in qk_parameter_items(model, "all")}
    other_name = next(
        name for name, parameter in named.items() if name not in qk_names and parameter in optimizer.state
    )
    other_before = {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in optimizer.state[named[other_name]].items()
    }

    alignments = iter([0.1, 0.4])
    losses = iter([1.00, 1.20, 0.50])
    monkeypatch.setattr("teleport.compute_qk_muon_alignment", lambda *args, **kwargs: next(alignments))
    monkeypatch.setattr("teleport.virtual_optimizer_step", lambda *args, **kwargs: next(losses))
    accepted, info = try_teleportation(
        model,
        optimizer,
        (images, labels),
        nn.CrossEntropyLoss(),
        teleport_args(tele_candidates=2),
        0,
    )
    assert accepted is True
    for name, parameter in qk_parameter_items(model, "all"):
        assert parameter not in optimizer.state or optimizer.state[parameter] == {}
    for key, value in other_before.items():
        live_value = optimizer.state[named[other_name]][key]
        if torch.is_tensor(value):
            assert torch.equal(live_value, value)
        else:
            assert live_value == value

    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(pixel_values=images).logits, labels)
    loss.backward()
    optimizer.step()
    for _, parameter in qk_parameter_items(model, "all"):
        group = next(group for group in optimizer.param_groups if any(parameter is item for item in group["params"]))
        if group.get("use_muon"):
            assert "momentum_buffer" in optimizer.state[parameter]


def test_plain_muon_and_naive_teleport_still_run():
    model = tiny_model()
    images, labels = tiny_batch(model)
    optimizer = create_muon_optimizer(model, lr=0.02, weight_decay=1e-4, momentum=0.95)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(pixel_values=images).logits, labels)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)

    naive_model = tiny_model()
    naive_args = teleport_args(tele_method="naive", n_teleport=3, tele_low=0.7, tele_high=1.4)
    try_naive_teleportation(
        naive_model,
        images,
        labels,
        criterion,
        naive_args,
        high=naive_args.tele_high,
        low=naive_args.tele_low,
        sign=0,
    )
    with torch.no_grad():
        logits = naive_model(pixel_values=images).logits
    assert torch.isfinite(logits).all()
