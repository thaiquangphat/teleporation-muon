import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for log_path in sorted(results_dir.rglob("log.json")):
        with log_path.open(encoding="utf-8") as log_file:
            log = json.load(log_file)

        epochs = log.get("epochs", [])
        if not epochs:
            continue

        config = log.get("config", {})
        dataset = config.get("dataset") or log.get("dataset", {}).get("name")
        if not dataset:
            print(f"Skipping log without dataset metadata: {log_path}")
            continue

        method = config.get("tele_method", "unknown")
        state_mode = config.get("tele_state_mode", "unknown")
        optimizer = config.get("opt", "unknown")
        runs.append(
            {
                "path": log_path,
                "dataset": dataset,
                "method": method,
                "state_mode": state_mode,
                "label": f"{optimizer} / {method} / {state_mode}",
                "epochs": epochs,
                "best_val_acc": log.get("best_val_acc"),
            }
        )
    return runs


def plot_dataset_metrics(dataset: str, runs: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for run in runs:
        records = run["epochs"]
        epoch_numbers = [record["epoch"] for record in records]
        axes[0].plot(
            epoch_numbers,
            [record["val_acc"] for record in records],
            marker="o",
            markersize=2,
            label=run["label"],
        )
        axes[1].plot(
            epoch_numbers,
            [record["train_loss"] for record in records],
            label=f'{run["label"]} train',
        )
        axes[1].plot(
            epoch_numbers,
            [record["val_loss"] for record in records],
            linestyle="--",
            alpha=0.8,
            label=f'{run["label"]} validation',
        )

    axes[0].set(title=f"{dataset}: validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set(title=f"{dataset}: loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / f"{dataset.lower()}_training_curves.png", dpi=160)
    plt.close(fig)


def plot_best_accuracy(runs: list[dict], output_dir: Path) -> None:
    labels = [f'{run["dataset"]}\n{run["label"]}' for run in runs]
    scores = []
    for run in runs:
        best = run["best_val_acc"]
        if best is None:
            best = max(record["val_acc"] for record in run["epochs"])
        scores.append(float(best))

    width = max(10, len(runs) * 0.65)
    fig, axis = plt.subplots(figsize=(width, 6))
    bars = axis.bar(range(len(runs)), scores)
    axis.set(
        title="Best validation accuracy by experiment",
        xlabel="Dataset / optimizer / method / state mode",
        ylabel="Best validation accuracy",
        ylim=(0, 1),
    )
    axis.set_xticks(range(len(runs)), labels, rotation=55, ha="right", fontsize="small")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.3f", padding=2, fontsize="x-small")
    fig.tight_layout()
    fig.savefig(output_dir / "best_validation_accuracy.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot validation accuracy and loss from MUONTELEPORTATION run logs."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise RuntimeError(f"No completed epoch logs found under {args.results_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted({run["dataset"] for run in runs}):
        plot_dataset_metrics(
            dataset,
            [run for run in runs if run["dataset"] == dataset],
            args.output_dir,
        )
    plot_best_accuracy(runs, args.output_dir)
    print(f"Plotted {len(runs)} runs into {args.output_dir}")


if __name__ == "__main__":
    main()
