"""Scoring criteria used by the ViT teleportation candidate search.

Each criterion exposes ``metrics_from_optimizer(optimizer)``. Candidate
generation and committing transformations are deliberately kept in
``teleport.py``.
"""

from .csd_l2 import csd_l2_metrics_from_optimizer
from .csd_l2_nesterov import nesterov_nuclear_metrics_from_optimizer
from .csd_l2_nesterov_frobenius import nesterov_frobenius_metrics_from_optimizer
from .csd_l2_nesterov_post_orth import post_orthogonalization_metrics_from_optimizer
from .csd_l2_nesterov_post_orth_frobenius import nesterov_post_orth_frobenius_metrics_from_optimizer

SCORING_CRITERIA = {
    "csd_l2": {
        "metrics_fn": csd_l2_metrics_from_optimizer,
        "method_name": "csd_l2",
        "score_definition": (
            "sqrt(sum_over_muon_matrices(nuclear_norm("
            "beta*previous_momentum + (1-beta)*current_gradient)^2))"
        ),
    },
    "csd_l2_nesterov": {
        "metrics_fn": nesterov_nuclear_metrics_from_optimizer,
        "method_name": "csd_l2_nesterov",
        "score_definition": (
            "sqrt(sum_over_muon_matrices(nuclear_norm("
            "beta^2*previous_momentum + (1-beta^2)*current_gradient)^2))"
        ),
    },
    "csd_l2_nesterov_frobenius": {
        "metrics_fn": nesterov_frobenius_metrics_from_optimizer,
        "method_name": "csd_l2_nesterov_frobenius",
        "score_definition": (
            "sqrt(sum_over_muon_matrices(frobenius_norm("
            "beta^2*previous_momentum + (1-beta^2)*current_gradient)^2))"
        ),
    },
    "csd_l2_nesterov_post_orth": {
        "metrics_fn": post_orthogonalization_metrics_from_optimizer,
        "method_name": "csd_l2_nesterov_post_orth",
        "score_definition": (
            "sqrt(sum_over_muon_matrices(nuclear_norm("
            "muon_newton_schulz(beta^2*previous_momentum + "
            "(1-beta^2)*current_gradient))^2))"
        ),
    },
    "csd_l2_nesterov_post_orth_frobenius": {
        "metrics_fn": nesterov_post_orth_frobenius_metrics_from_optimizer,
        "method_name": "csd_l2_nesterov_post_orth_frobenius",
        "score_definition": (
            "sqrt(sum_over_muon_matrices(frobenius_norm("
            "muon_newton_schulz(beta^2*previous_momentum + "
            "(1-beta^2)*current_gradient))^2))"
        ),
    },
}

def get_scoring_criterion(method_name):
    """Resolve a scoring method, preserving the historical alias."""
    canonical_name = "csd_l2" if method_name == "muon_alignment" else method_name
    try:
        return SCORING_CRITERIA[canonical_name]
    except KeyError as exc:
        supported = ", ".join(sorted(SCORING_CRITERIA))
        raise ValueError(
            f"Unknown scoring criterion {method_name!r}. Supported criteria: {supported}."
        ) from exc

__all__ = ["SCORING_CRITERIA", "get_scoring_criterion"]
