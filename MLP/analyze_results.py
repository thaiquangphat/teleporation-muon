import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def mean(values):
    return statistics.mean(values) if values else None


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def read_runs(results_dir):
    runs = []
    for log_path in sorted(results_dir.rglob("log.json")):
        try:
            with log_path.open(encoding="utf-8") as log_file:
                data = json.load(log_file)
        except (OSError, json.JSONDecodeError) as error:
            print(f"Warning: cannot read {log_path}: {error}", file=sys.stderr)
            continue

        config = data.get("config", {})
        dataset = data.get("dataset", {}).get("name", config.get("dataset", "unknown"))
        optimizer = config.get("opt", "unknown")
        teleported = int(config.get("n_teleport", 0)) > 0
        epochs = data.get("epochs", [])
        train_seconds = sum(epoch.get("train_time_seconds", 0.0) for epoch in epochs)
        validation_seconds = sum(
            epoch.get("validation_time_seconds", 0.0) for epoch in epochs
        )
        runs.append(
            {
                "path": str(log_path),
                "run_id": data.get("run_id"),
                "run_name": data.get("run_name"),
                "status": data.get("status", "unknown"),
                "dataset": dataset,
                "optimizer": optimizer,
                "teleported": teleported,
                "seed": config.get("seed"),
                "epochs_completed": len(epochs),
                "best_val_acc": data.get("best_val_acc"),
                "best_epoch": data.get("best_epoch"),
                "total_training_time_seconds": data.get(
                    "total_training_time_seconds"
                ),
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "mean_epoch_seconds": mean(
                    [epoch.get("epoch_time_seconds", 0.0) for epoch in epochs]
                ),
                "epochs": epochs,
            }
        )
    return runs


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def summarize(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["dataset"], run["optimizer"], run["teleported"])].append(run)

    summaries = []
    for (dataset, optimizer, teleported), group_runs in sorted(groups.items()):
        successful = [run for run in group_runs if run["status"] == "completed"]
        accuracies = [
            run["best_val_acc"]
            for run in successful
            if run["best_val_acc"] is not None
        ]
        wall_times = [
            run["total_training_time_seconds"]
            for run in successful
            if run["total_training_time_seconds"] is not None
        ]
        train_times = [run["train_seconds"] for run in successful]
        validation_times = [run["validation_seconds"] for run in successful]
        summaries.append(
            {
                "dataset": dataset,
                "optimizer": optimizer,
                "teleported": teleported,
                "runs": len(group_runs),
                "successful_runs": len(successful),
                "seeds": ",".join(
                    str(run["seed"])
                    for run in sorted(group_runs, key=lambda item: str(item["seed"]))
                ),
                "best_val_acc_mean": mean(accuracies),
                "best_val_acc_std": stdev(accuracies),
                "total_training_time_seconds_mean": mean(wall_times),
                "total_training_time_seconds_std": stdev(wall_times),
                "train_seconds_mean": mean(train_times),
                "validation_seconds_mean": mean(validation_times),
            }
        )
    return summaries


def plot_curves(runs, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with `python -m pip install matplotlib`."
        ) from error

    curve_groups = defaultdict(list)
    for run in runs:
        if run["status"] == "completed":
            curve_groups[
                (run["dataset"], run["optimizer"], run["teleported"])
            ].append(run)

    for metric, title, ylabel, filename in (
        ("val_acc", "Validation accuracy across seeds", "Accuracy", "validation_accuracy.png"),
        ("train_loss", "Training loss across seeds", "Loss", "training_loss.png"),
        ("val_loss", "Validation loss across seeds", "Loss", "validation_loss.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
        axes = axes[0]
        for axis, dataset in zip(axes, ("MNIST", "CIFAR10")):
            for (group_dataset, optimizer, teleported), group_runs in sorted(
                curve_groups.items()
            ):
                if group_dataset != dataset:
                    continue
                per_epoch = defaultdict(list)
                for run in group_runs:
                    for epoch in run["epochs"]:
                        if metric in epoch:
                            per_epoch[epoch["epoch"]].append(epoch[metric])
                epochs = sorted(per_epoch)
                if not epochs:
                    continue
                values = [mean(per_epoch[epoch]) for epoch in epochs]
                spreads = [stdev(per_epoch[epoch]) or 0.0 for epoch in epochs]
                label = f"{optimizer}, {'teleport' if teleported else 'baseline'}"
                axis.plot(epochs, values, marker="o", label=label)
                axis.fill_between(
                    epochs,
                    [value - spread for value, spread in zip(values, spreads)],
                    [value + spread for value, spread in zip(values, spreads)],
                    alpha=0.15,
                )
            axis.set_title(dataset)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize="small")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    completed = [run for run in runs if run["status"] == "completed"]
    groups = defaultdict(list)
    for run in completed:
        if run["total_training_time_seconds"] is not None:
            groups[(run["dataset"], run["optimizer"], run["teleported"])].append(
                run["total_training_time_seconds"]
            )

    labels = []
    values = []
    errors = []
    for key, times in sorted(groups.items()):
        labels.append(
            f"{key[0]}\n{key[1]}\n{'teleport' if key[2] else 'baseline'}"
        )
        values.append(mean(times))
        errors.append(stdev(times) or 0.0)
    if labels:
        fig, axis = plt.subplots(figsize=(max(10, len(labels) * 1.15), 6))
        axis.bar(range(len(labels)), values, yerr=errors, capsize=4)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_ylabel("Training wall time (seconds)")
        axis.set_title("Mean total training time across runs (error bars: standard deviation)")
        axis.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "training_time.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate MLP training logs and plot seed-averaged results."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        parser.error(f"Results directory does not exist: {args.results_dir}")
    runs = read_runs(args.results_dir)
    if not runs:
        parser.error(f"No log.json files found under {args.results_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_fields = [
        "path",
        "run_id",
        "run_name",
        "status",
        "dataset",
        "optimizer",
        "teleported",
        "seed",
        "epochs_completed",
        "best_val_acc",
        "best_epoch",
        "total_training_time_seconds",
        "train_seconds",
        "validation_seconds",
        "mean_epoch_seconds",
    ]
    summary_fields = [
        "dataset",
        "optimizer",
        "teleported",
        "runs",
        "successful_runs",
        "seeds",
        "best_val_acc_mean",
        "best_val_acc_std",
        "total_training_time_seconds_mean",
        "total_training_time_seconds_std",
        "train_seconds_mean",
        "validation_seconds_mean",
    ]
    write_csv(args.output_dir / "runs.csv", runs, run_fields)
    summaries = summarize(runs)
    write_csv(args.output_dir / "summary.csv", summaries, summary_fields)
    plot_curves(runs, args.output_dir)

    print(f"Read {len(runs)} run logs.")
    print(f"Per-run metrics: {args.output_dir / 'runs.csv'}")
    print(f"Seed-group summary: {args.output_dir / 'summary.csv'}")
    print(f"Plots: {args.output_dir}")


if __name__ == "__main__":
    main()
