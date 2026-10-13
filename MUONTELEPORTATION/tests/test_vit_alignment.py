import copy

import pytest
import torch
from torch import nn
from torch.optim import AdamW, SGD

from src.model import ViTConfig, ViTForImageClassification, build_vit
from src.muon import create_muon_optimizer
from src.vit_alignment import (
    apply_attention_teleportation,
    generate_attention_scales,
    reset_attention_optimizer_state,
    try_vit_alignment_teleportation,
)


@pytest.fixture
def model():
    config = ViTConfig(
        image_size=8,
        num_channels=1,
        num_classes=3,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        patch_size=4,
    )
    return ViTForImageClassification(config)


@pytest.mark.parametrize(("image_size", "num_channels"), [(28, 1), (32, 3)])
def test_build_vit_accepts_mnist_and_cifar_shapes(image_size, num_channels):
    model = build_vit(
        image_size=image_size,
        num_channels=num_channels,
        num_classes=10,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        patch_size=4,
    )
    images = torch.randn(2, num_channels, image_size, image_size)
    assert model(pixel_values=images).logits.shape == (2, 10)


def test_attention_transform_preserves_logits(model):
    images = torch.randn(2, 1, 8, 8)
    model.eval()
    with torch.no_grad():
        original = model(pixel_values=images).logits
        apply_attention_teleportation(
            model,
            torch.tensor([0.7, 0.7, 0.85, 0.85] * 4),
        )
        transformed = model(pixel_values=images).logits
    assert torch.allclose(original, transformed, atol=2e-6, rtol=2e-6)


def test_alignment_teleportation_scores_muon_on_first_step(model):
    images = torch.randn(2, 1, 8, 8)
    labels = torch.tensor([0, 2])
    optimizer = create_muon_optimizer(model, lr=0.01, weight_decay=1e-4, momentum=0.9)
    accepted, info = try_vit_alignment_teleportation(
        model,
        optimizer,
        (images, labels),
        nn.CrossEntropyLoss(),
        low=0.8,
        high=1.2,
        candidate_count=2,
        accept_tolerance=0.0,
        seed=11,
    )
    assert accepted == (info["selected_alignment"] > info["identity_alignment"])
    assert len(info["candidate_scores"]) == 2
    assert all(torch.isfinite(torch.tensor(item["alignment"])) for item in info["candidate_scores"])
    assert info["state_mode"] == "reset"
    if accepted:
        layer = model.vit.encoder.layer[0].attention.attention
        assert layer.query.weight not in optimizer.state
        assert layer.key.weight not in optimizer.state


@pytest.mark.parametrize("optimizer_name", ["adamw", "sgd"])
def test_alignment_teleportation_supports_baseline_optimizers(model, optimizer_name):
    images = torch.randn(2, 1, 8, 8)
    labels = torch.tensor([0, 2])
    if optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=1e-3)
    else:
        optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9)
    _, info = try_vit_alignment_teleportation(
        model,
        optimizer,
        (images, labels),
        nn.CrossEntropyLoss(),
        low=0.8,
        high=1.2,
        candidate_count=2,
        accept_tolerance=1e30,
        seed=4,
    )
    assert len(info["candidate_scores"]) == 2
    assert all(torch.isfinite(torch.tensor(item["alignment"])) for item in info["candidate_scores"])


def test_virtual_alignment_ignores_stale_live_gradients(model):
    stale_gradient_model = copy.deepcopy(model)
    for parameter in stale_gradient_model.parameters():
        parameter.grad = torch.ones_like(parameter)

    images = torch.randn(2, 1, 8, 8)
    labels = torch.tensor([0, 2])
    clean_optimizer = AdamW(model.parameters(), lr=1e-3)
    stale_optimizer = AdamW(stale_gradient_model.parameters(), lr=1e-3)
    options = {
        "low": 0.8,
        "high": 1.2,
        "candidate_count": 2,
        "accept_tolerance": 1e30,
        "seed": 17,
    }
    _, clean_info = try_vit_alignment_teleportation(
        model, clean_optimizer, (images, labels), nn.CrossEntropyLoss(), **options
    )
    _, stale_info = try_vit_alignment_teleportation(
        stale_gradient_model,
        stale_optimizer,
        (images, labels),
        nn.CrossEntropyLoss(),
        **options,
    )
    assert [item["alignment"] for item in clean_info["candidate_scores"]] == pytest.approx(
        [item["alignment"] for item in stale_info["candidate_scores"]]
    )


def test_reset_clears_only_attention_query_key_state(model):
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    for parameter in model.parameters():
        optimizer.state[parameter]["momentum_buffer"] = torch.ones_like(parameter)
    first_layer = model.vit.encoder.layer[0]
    unrelated_parameter = model.classifier.weight

    reset_attention_optimizer_state(model, optimizer)

    assert first_layer.attention.attention.query.weight not in optimizer.state
    assert first_layer.attention.attention.key.bias not in optimizer.state
    assert unrelated_parameter in optimizer.state


def test_scale_candidates_include_identity_and_respect_bounds():
    candidates = generate_attention_scales(
        8, 0.7, 1.4, 4, seed=9, position_embedding="rope", num_heads=2
    )
    assert torch.equal(candidates[0], torch.ones(8))
    for candidate in candidates[1:]:
        assert torch.all(candidate >= 0.7)
        assert torch.all(candidate <= 1.4)
        assert torch.equal(candidate.reshape(2, 2, 2)[..., 0], candidate.reshape(2, 2, 2)[..., 1])


def test_rope_rejects_scales_that_break_rotary_pairs(model):
    with pytest.raises(ValueError, match="equal within each rotary channel pair"):
        apply_attention_teleportation(model, torch.tensor([0.5, 1.0] * 8))


def test_vit_training_step_runs_with_muon(model):
    images = torch.randn(2, 1, 8, 8)
    labels = torch.tensor([0, 2])
    optimizer = create_muon_optimizer(model, lr=0.01, weight_decay=1e-4, momentum=0.9)
    loss = nn.CrossEntropyLoss()(model(pixel_values=images).logits, labels)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
