import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(project: str, entity: str | None = None) -> list[dict]:
    api = wandb.Api()
    entity = entity or api.default_entity
    if not entity:
        raise RuntimeError("Set WANDB_ENTITY or configure a default W&B entity.")

    runs = []
    for run in api.runs(f"{entity}/{project}"):
        config = dict(run.config)
        history_keys = [
            "epoch",
            "train/epoch_loss",
            "validation/loss",
            "validation/accuracy",
        ]
        epochs = [
            {
                "epoch": int(row["epoch"]),
                "train_loss": row["train/epoch_loss"],
                "val_loss": row["validation/loss"],
                "val_acc": row["validation/accuracy"],
            }
            for row in run.scan_history(keys=history_keys)
            if row.get("epoch") is not None
            and row.get("train/epoch_loss") is not None
            and row.get("validation/loss") is not None
            and row.get("validation/accuracy") is not None
        ]
        if not epochs:
            continue

        dataset = config.get("dataset")
        if not dataset:
            continue
        if run.state != "finished":
            continue

        method = config.get("tele_method", "unknown")
        state_mode = config.get("tele_state_mode", "unknown")
        optimizer = config.get("opt", "unknown")
        runs.append(
            {
                "path": run.url,
                "dataset": dataset,
                "method": method,
                "state_mode": state_mode,
                "label": f"{optimizer} / {method} / {state_mode}",
                "config": config,
                "seed": config.get("seed"),
                "epochs": epochs,
                "best_val_acc": run.summary.get("best/validation_accuracy"),
            }
        )
    return runs


def average_runs(runs: list[dict]) -> list[dict]:
    grouped_runs: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        config = {key: value for key, value in run["config"].items() if key != "seed"}
        group_key = json.dumps(config, sort_keys=True)
        grouped_runs[group_key].append(run)

    averages = []
    for group in grouped_runs.values():
        seeds = [run["seed"] for run in group]
        if len(set(seeds)) != len(seeds):
            raise RuntimeError(f"Duplicate seed logs found for experiment: {group[0]['label']}")

        epoch_numbers = [
            {record["epoch"] for record in run["epochs"]}
            for run in group
        ]
        if any(numbers != epoch_numbers[0] for numbers in epoch_numbers[1:]):
            raise RuntimeError(
                f"Epochs differ between seeds for experiment: {group[0]['label']}"
            )

        averaged_epochs = []
        for epoch in sorted(epoch_numbers[0]):
            records = [
                next(record for record in run["epochs"] if record["epoch"] == epoch)
                for run in group
            ]
            averaged_epochs.append(
                {
                    "epoch": epoch,
                    "train_loss": mean(record["train_loss"] for record in records),
                    "val_loss": mean(record["val_loss"] for record in records),
                    "val_acc": mean(record["val_acc"] for record in records),
                }
            )

        best_scores = [
            run["best_val_acc"]
            if run["best_val_acc"] is not None
            else max(record["val_acc"] for record in run["epochs"])
            for run in group
        ]
        average = dict(group[0])
        average["epochs"] = averaged_epochs
        average["best_val_acc"] = mean(best_scores)
        average["seed_count"] = len(group)
        average["label"] = f'{group[0]["label"]} (mean of {len(group)} seeds)'
        averages.append(average)
    return averages


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
        description="Plot mean validation accuracy and loss from W&B experiment runs."
    )
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.project:
        raise RuntimeError("Set WANDB_PROJECT or pass --project.")

    runs = average_runs(load_runs(args.project, args.entity))
    if not runs:
        raise RuntimeError(f"No completed epoch logs found in W&B project {args.project}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted({run["dataset"] for run in runs}):
        plot_dataset_metrics(
            dataset,
            [run for run in runs if run["dataset"] == dataset],
            args.output_dir,
        )
    plot_best_accuracy(runs, args.output_dir)
    print(
        f"Plotted {len(runs)} experiment means from "
        f"{sum(run['seed_count'] for run in runs)} seed runs into {args.output_dir}"
    )


if __name__ == "__main__":
    main()
