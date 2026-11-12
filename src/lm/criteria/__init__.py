"""Scoring criteria used by the LM teleportation candidate search.

Each criterion exposes a metrics function. Candidate construction, evaluation,
ranking, and commitment are deliberately kept in ``src/lm/teleport.py``,
matching the ViT implementation.
"""

from .csd_l2 import csd_l2_metrics_from_optimizer
from .csd_l2_nesterov import (
    nesterov_csd_l2_metrics,
    nesterov_nuclear_metrics_from_optimizer,
)
from .csd_l2_nesterov_frobenius import nesterov_frobenius_metrics_from_optimizer
from .csd_l2_nesterov_post_orth import post_orthogonalization_metrics_from_optimizer
from .csd_l2_nesterov_post_orth_frobenius import (
    nesterov_post_orth_frobenius_metrics_from_optimizer,
)


SCORING_CRITERIA = {
    "csd_l2": {
        "metrics_fn": csd_l2_metrics_from_optimizer,
        "method_name": "csd_l2",
        "requires_nesterov": False,
        "score_definition": (
            "sqrt(sum_over_muon_matrices(nuclear_norm("
            "beta*previous_momentum + (1-beta)*current_gradient)^2))"
        ),
    },
    "csd_l2_nesterov": {
        "metrics_fn": nesterov_nuclear_metrics_from_optimizer,
        "method_name": "csd_l2_nesterov",
        "requires_nesterov": True,
        "score_definition": (
            "sqrt(sum_over_muon_matrices(nuclear_norm("
            "beta^2*previous_momentum + (1-beta^2)*current_gradient)^2))"
        ),
    },
    "csd_l2_nesterov_frobenius": {
        "metrics_fn": nesterov_frobenius_metrics_from_optimizer,
        "method_name": "csd_l2_nesterov_frobenius",
        "requires_nesterov": True,
        "score_definition": (
            "sqrt(sum_over_muon_matrices(frobenius_norm("
            "beta^2*previous_momentum + (1-beta^2)*current_gradient)^2))"
        ),
    },
    "csd_l2_nesterov_post_orth": {
        "metrics_fn": post_orthogonalization_metrics_from_optimizer,
        "method_name": "csd_l2_nesterov_post_orth",
        "requires_nesterov": True,
        "score_definition": (
            "sqrt(sum_over_muon_matrices(nuclear_norm("
            "muon_newton_schulz(beta^2*previous_momentum + "
            "(1-beta^2)*current_gradient))^2))"
        ),
    },
    "csd_l2_nesterov_post_orth_frobenius": {
        "metrics_fn": nesterov_post_orth_frobenius_metrics_from_optimizer,
        "method_name": "csd_l2_nesterov_post_orth_frobenius",
        "requires_nesterov": True,
        "score_definition": (
            "sqrt(sum_over_muon_matrices(frobenius_norm("
            "muon_newton_schulz(beta^2*previous_momentum + "
            "(1-beta^2)*current_gradient))^2))"
        ),
    },
}


def get_scoring_criterion(method_name):
    """Resolve a registered score definition."""
    canonical_name = "csd_l2_nesterov" if method_name == "muon_alignment" else method_name
    try:
        return SCORING_CRITERIA[canonical_name]
    except KeyError as exc:
        supported = ", ".join(sorted(SCORING_CRITERIA))
        raise ValueError(
            f"Unknown scoring criterion {method_name!r}. Supported criteria: {supported}."
        ) from exc


__all__ = [
    "SCORING_CRITERIA",
    "get_scoring_criterion",
    "csd_l2_metrics_from_optimizer",
    "nesterov_csd_l2_metrics",
    "nesterov_nuclear_metrics_from_optimizer",
    "nesterov_frobenius_metrics_from_optimizer",
    "post_orthogonalization_metrics_from_optimizer",
    "nesterov_post_orth_frobenius_metrics_from_optimizer",
]
