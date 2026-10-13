import pytest

from plot_results import average_runs


def make_run(seed, train_loss, val_loss, val_acc, best_val_acc):
    return {
        "config": {"dataset": "MNIST", "opt": "muon", "seed": seed},
        "seed": seed,
        "dataset": "MNIST",
        "method": "none",
        "state_mode": "unknown",
        "label": "muon / none / unknown",
        "epochs": [
            {
                "epoch": 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        ],
        "best_val_acc": best_val_acc,
    }


def test_average_runs_averages_metrics_across_expected_seeds():
    runs = [
        make_run(1, 0.3, 0.5, 0.6, 0.7),
        make_run(2, 0.2, 0.4, 0.7, 0.8),
        make_run(3, 0.1, 0.3, 0.8, 0.9),
    ]

    [average] = average_runs(runs)

    assert average["seed_count"] == 3
    assert average["epochs"] == [
        {
            "epoch": 1,
            "train_loss": pytest.approx(0.2),
            "val_loss": pytest.approx(0.4),
            "val_acc": pytest.approx(0.7),
        }
    ]
    assert average["best_val_acc"] == pytest.approx(0.8)


def test_average_runs_rejects_duplicate_seed_logs():
    runs = [make_run(1, 0.3, 0.5, 0.6, 0.7), make_run(1, 0.2, 0.4, 0.7, 0.8)]

    with pytest.raises(RuntimeError, match="Duplicate seed logs"):
        average_runs(runs)
