import argparse
import os
import time

import torch
import wandb
import yaml
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification

from datamodule import get_dataset
from model import ViTRoPEForImageClassification
from muon import create_muon_optimizer
from teleport import generate_tele_scheduler, try_teleportation


def list_of_ints(arg):
    return [int(x) for x in arg.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT Teleportation")

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, choices=["CIFAR100", "CIFAR10", "MNIST", "IMAGENET"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--devices", type=str, default="cuda:0")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-hidden-layer", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--position-embedding", type=str, default="learnable", choices=["rope", "learnable"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--opt", type=str, default="adamw", choices=["adamw", "sgd", "muon"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["step", "cosine", "plateau"])
    parser.add_argument("--n-teleport", type=int, default=0, help="number of naive teleport candidates")
    parser.add_argument("--tele-batch", type=int, default=32, help="Number of batch per Tele")
    parser.add_argument("--tele-epoch-array", type=list_of_ints, default=[1], help="Epoch of teleports")
    parser.add_argument("--tele-att", type=int, default=1, help="Teleport Attention")
    parser.add_argument("--tele-mlp", type=int, default=0, help="Teleport MLP")
    parser.add_argument("--tele-opt", type=int, default=1, help="Teleport Option")
    parser.add_argument("--tele-high", type=float, default=1.1, help="Teleport Max Entries")
    parser.add_argument("--tele-low", type=float, default=0.9, help="Teleport Min Entries")
    parser.add_argument("--tele-cons", type=int, default=16, help="Consecutive")
    parser.add_argument("--tele-sign", type=int, default=0, help="Sign")
    parser.add_argument("--tele-layer", type=str, default="all", choices=["all", "first", "last"])
    parser.add_argument(
        "--tele-method",
        type=str,
        default="none",
        choices=["none", "naive", "muon_alignment"],
        help="none keeps a plain optimizer baseline; naive is the original MNISTCIFAR teleport; muon_alignment is the Q/K Muon-aware search.",
    )
    parser.add_argument("--tele-candidates", type=int, default=6)
    parser.add_argument("--tele-accept-tol", type=float, default=1e-6)
    parser.add_argument("--max-train-batches", type=int, default=0, help="If > 0, stop each epoch after this many train batches.")
    return parser.parse_args()


def build_model(args, device):
    config = ViTConfig(
        hidden_size=args.d_model,
        num_hidden_layers=args.num_hidden_layer,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        image_size=args.img_size,
        patch_size=args.patch_size,
        num_channels=args.num_channels,
        num_labels=args.num_classes,
        qkv_bias=True,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    if args.position_embedding == "rope":
        model = ViTRoPEForImageClassification(config)
    else:
        model = ViTForImageClassification(config)
    return model.to(device)


def build_optimizer(model, args):
    if args.opt == "adamw":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.opt == "sgd":
        return SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    return create_muon_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )


def build_run_name(args):
    if args.tele_method == "none" and args.n_teleport <= 0:
        return (
            f"NoTeleport-lr{args.lr}_batch{args.batch_size}_{args.opt}_embed{args.position_embedding}_seed{args.seed}"
        )
    return (
        f"Teleport-{args.tele_method}-lr{args.lr}_batch{args.batch_size}_embeded{args.position_embedding}_opt{args.opt}"
        f"att{args.tele_att}_mlp{args.tele_mlp}_"
        f"tele-opt{args.tele_opt}_tele-high{args.tele_high}_tele-low{args.tele_low}_tele-cons{args.tele_cons}_tele-sign{args.tele_sign}_"
        f"tele-layer{args.tele_layer}_tele-batch{args.tele_batch}_tele-epoch{args.tele_epoch_array}_cands{args.tele_candidates}_seed{args.seed}"
    )


def maybe_teleport(args):
    return args.tele_method != "none" or args.n_teleport > 0


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_wandb_settings(args):
    config_path = os.environ.get("WANDB_CONFIG_PATH", "./wandb.yaml")
    file_config = {}
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as config_file:
            file_config = yaml.safe_load(config_file) or {}
        if not isinstance(file_config, dict):
            raise ValueError(f"W&B config must be a YAML mapping: {config_path}")

    project = args.wandb_project or os.environ.get("WANDB_PROJECT") or file_config.get("project")
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or file_config.get("entity") or None
    mode = args.wandb_mode or os.environ.get("WANDB_MODE") or file_config.get("mode") or "online"
    api_key = os.environ.get("WANDB_API_KEY") or file_config.get("api_key")

    if not project:
        raise ValueError("Set project in wandb.yaml, WANDB_PROJECT, or --wandb-project.")
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("W&B mode must be online, offline, or disabled.")
    if mode == "online" and not api_key:
        raise ValueError("Set api_key in wandb.yaml or WANDB_API_KEY for online W&B.")
    if api_key:
        os.environ["WANDB_API_KEY"] = str(api_key)
    return str(project), str(entity) if entity else None, str(mode)


def main():
    args = parse_args()
    run_started = time.perf_counter()
    if args.tele_method == "naive" and args.n_teleport <= 0:
        args.n_teleport = max(args.tele_candidates, 2)

    torch.manual_seed(args.seed)
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")

    run_name = build_run_name(args)
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    setup_started = time.perf_counter()
    data_started = time.perf_counter()
    train_loader, val_loader = get_dataset(args)
    data_loading_time_seconds = time.perf_counter() - data_started
    model_started = time.perf_counter()
    model = build_model(args, device)
    optimizer = build_optimizer(model, args)
    model_setup_time_seconds = time.perf_counter() - model_started

    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    else:
        scheduler = None

    tele_scheduler = None
    if maybe_teleport(args):
        tele_scheduler = generate_tele_scheduler(
            tele_batch=args.tele_batch,
            number_of_batch=len(train_loader),
            tele_opt=args.tele_opt,
            tele_cons=args.tele_cons,
        )

    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_acc = 0.0
    best_epoch = 0
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    project, entity, mode = load_wandb_settings(args)
    with wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=vars(args),
        dir=save_dir,
        mode=mode,
        job_type="train",
    ) as run:
        run_started = time.perf_counter()
        run.config.update(
            {
                "environment_device": str(device),
                "environment_gpu_name": gpu_name or "cpu",
                "environment_torch_version": torch.__version__,
                "environment_cuda_version": torch.version.cuda or "none",
                "dataset_train_samples": len(train_loader.dataset),
                "dataset_validation_samples": len(val_loader.dataset),
                "dataset_train_batches_per_epoch": len(train_loader),
                "dataset_validation_batches_per_epoch": len(val_loader),
                "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            },
            allow_val_change=True,
        )
        run.log(
            {
                "timing/data_loading_seconds": data_loading_time_seconds,
                "timing/model_setup_seconds": model_setup_time_seconds,
                "timing/initialization_seconds": time.perf_counter() - setup_started,
            },
        )
        attempted_teleports = 0
        accepted_teleports = 0
        total_train_time_seconds = 0.0
        total_validation_time_seconds = 0.0
        total_checkpoint_time_seconds = 0.0
        telemetry_table = wandb.Table(
            columns=[
                "epoch",
                "batch",
                "global_step",
                "method",
                "accepted",
                "reason",
                "search_time_seconds",
                "alignment_gain",
                "selected_alignment",
                "selected_candidate_index",
                "num_candidates_evaluated",
            ]
        )
        candidate_table = wandb.Table(
            columns=[
                "epoch",
                "batch",
                "global_step",
                "method",
                "candidate_index",
                "is_identity",
                "alignment",
                "alignment_rank",
                "post_step_loss_evaluated",
                "post_step_loss",
            ]
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1, args.epochs + 1):
            synchronize_device(device)
            epoch_started = time.perf_counter()
            model.train()
            running_loss = 0.0
            processed_batches = 0
            processed_samples = 0
            for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Training")):
                if args.max_train_batches > 0 and i >= args.max_train_batches:
                    break
                images, labels = images.to(device), labels.to(device)

                if (
                    maybe_teleport(args)
                    and tele_scheduler is not None
                    and tele_scheduler[i]
                    and epoch in args.tele_epoch_array
                ):
                    method = args.tele_method
                    if method == "none" and args.n_teleport > 0:
                        method = "naive"
                    teleport_args = argparse.Namespace(**vars(args))
                    teleport_args.tele_method = method
                    accepted, info = try_teleportation(
                        model=model,
                        optimizer=optimizer,
                        batch=(images, labels),
                        criterion=lambda logits, target: criterion(logits, target),
                        args=teleport_args,
                        step_index=global_step,
                    )
                    info["epoch"] = epoch
                    info["batch"] = i
                    info["accepted"] = accepted
                    attempted_teleports += 1
                    accepted_teleports += int(accepted)
                    telemetry_table.add_data(
                        epoch,
                        i,
                        global_step,
                        info.get("method", method),
                        accepted,
                        info.get("reason", ""),
                        info.get("search_time_seconds", 0.0),
                        info.get("alignment_gain"),
                        info.get("selected_alignment"),
                        info.get("selected_candidate_index"),
                        info.get("num_candidates_evaluated"),
                    )
                    for candidate in info.get("results", []):
                        candidate_table.add_data(
                            epoch,
                            i,
                            global_step,
                            info.get("method", method),
                            candidate.get("candidate_index"),
                            candidate.get("identity"),
                            candidate.get("alignment"),
                            candidate.get("alignment_rank"),
                            candidate.get("post_step_loss") is not None,
                            candidate.get("post_step_loss"),
                        )
                    event_metrics = {
                        "teleport/attempted": attempted_teleports,
                        "teleport/accepted": accepted_teleports,
                        "teleport/acceptance_rate": accepted_teleports / attempted_teleports,
                        "teleport/search_time_seconds": info.get("search_time_seconds", 0.0),
                        "teleport/alignment_gain": info.get("alignment_gain"),
                        "teleport/selected_alignment": info.get("selected_alignment"),
                        "teleport/selected_candidate_index": info.get("selected_candidate_index"),
                        "teleport/num_candidates_evaluated": info.get("num_candidates_evaluated"),
                        "teleport/accepted_event": int(accepted),
                    }
                    run.log({key: value for key, value in event_metrics.items() if value is not None})

                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
                global_step += 1
                processed_batches += 1
                processed_samples += labels.size(0)
                run.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/batch_size": labels.size(0),
                    }
                )

            synchronize_device(device)
            train_time_seconds = time.perf_counter() - epoch_started
            total_train_time_seconds += train_time_seconds
            avg_loss = running_loss / max(processed_batches, 1)
            current_lr = optimizer.param_groups[0]["lr"]

            synchronize_device(device)
            validation_started = time.perf_counter()
            model.eval()
            correct, total = 0, 0
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(pixel_values=images).logits
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            synchronize_device(device)
            validation_time_seconds = time.perf_counter() - validation_started
            total_validation_time_seconds += validation_time_seconds
            avg_val_acc = correct / total
            avg_val_loss = val_loss / len(val_loader)

            checkpoint_started = time.perf_counter()
            last_ckpt_path = os.path.join(save_dir, "last.pt")
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "val_acc": avg_val_acc,
                "lr": current_lr,
            }
            torch.save(checkpoint, last_ckpt_path)

            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                best_epoch = epoch
                best_ckpt_path = os.path.join(save_dir, "best.pt")
                torch.save(checkpoint, best_ckpt_path)
            checkpoint_time_seconds = time.perf_counter() - checkpoint_started
            total_checkpoint_time_seconds += checkpoint_time_seconds

            scheduler_started = time.perf_counter()
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_acc)
            elif scheduler is not None:
                scheduler.step()
            scheduler_time_seconds = time.perf_counter() - scheduler_started

            epoch_time_seconds = time.perf_counter() - epoch_started
            gpu_memory_allocated = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda" else 0.0
            )
            run.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": avg_loss,
                    "train/learning_rate": current_lr,
                    "train/batches": processed_batches,
                    "train/samples": processed_samples,
                    "train/samples_per_second": (
                        processed_samples / train_time_seconds
                        if train_time_seconds > 0 else 0.0
                    ),
                    "validation/loss": avg_val_loss,
                    "validation/accuracy": avg_val_acc,
                    "validation/correct": correct,
                    "validation/samples": total,
                    "validation/samples_per_second": (
                        total / validation_time_seconds if validation_time_seconds > 0 else 0.0
                    ),
                    "best/validation_accuracy": best_acc,
                    "best/epoch": best_epoch,
                    "timing/train_seconds": train_time_seconds,
                    "timing/validation_seconds": validation_time_seconds,
                    "timing/evaluation_seconds": validation_time_seconds,
                    "timing/checkpoint_seconds": checkpoint_time_seconds,
                    "timing/scheduler_seconds": scheduler_time_seconds,
                    "timing/epoch_seconds": epoch_time_seconds,
                    "timing/total_training_seconds": total_train_time_seconds,
                    "timing/total_validation_seconds": total_validation_time_seconds,
                    "timing/total_evaluation_seconds": total_validation_time_seconds,
                    "timing/total_checkpoint_seconds": total_checkpoint_time_seconds,
                    "teleport/attempted": attempted_teleports,
                    "teleport/accepted": accepted_teleports,
                    "teleport/acceptance_rate": (
                        accepted_teleports / attempted_teleports if attempted_teleports else 0.0
                    ),
                    "system/gpu_peak_memory_mb": gpu_memory_allocated,
                    "global_step": global_step,
                }
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        run.log({"teleport/events": telemetry_table})
        run.log({"teleport/candidates": candidate_table})
        run.summary["best/validation_accuracy"] = best_acc
        run.summary["best/epoch"] = best_epoch
        run.summary["timing/total_training_seconds"] = total_train_time_seconds
        run.summary["timing/total_validation_seconds"] = total_validation_time_seconds
        run.summary["timing/total_evaluation_seconds"] = total_validation_time_seconds
        run.summary["timing/total_checkpoint_seconds"] = total_checkpoint_time_seconds
        run.summary["timing/run_seconds"] = time.perf_counter() - run_started
        run.summary["teleport/attempted"] = attempted_teleports
        run.summary["teleport/accepted"] = accepted_teleports


if __name__ == "__main__":
    main()
