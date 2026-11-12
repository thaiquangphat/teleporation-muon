import argparse
import os

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
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the main optimizer group (Muon when --opt=muon).")
    parser.add_argument("--adam-lr", type=float, default=3e-4, help="Learning rate for the auxiliary AdamW group when --opt=muon.")
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
        choices=[
            "none", "naive", "csd_l2", "csd_l2_nesterov",
            "csd_l2_nesterov_frobenius", "csd_l2_nesterov_post_orth", "csd_l2_nesterov_post_orth_frobenius",
            "muon_alignment",
        ],
        help=(
            "none keeps a plain optimizer baseline; naive is the original MNISTCIFAR teleport; "
            "csd_l2 scores EMA momentum; csd_l2_nesterov scores pre-orthogonalization nuclear "
            "norms; csd_l2_nesterov_frobenius scores pre-orthogonalization Frobenius norms; "
            "csd_l2_nesterov_post_orth scores after Muon's Newton--Schulz orthogonalization; "
            "muon_alignment aliases csd_l2."
        ),
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
        adam_lr=args.adam_lr,
    )


def build_run_name(args):
    adam_lr_suffix = f"_adam-lr{args.adam_lr}" if args.opt == "muon" else ""
    if args.tele_method == "none" and args.n_teleport <= 0:
        return (
            f"NoTeleport-lr{args.lr}{adam_lr_suffix}_batch{args.batch_size}_{args.opt}_embed{args.position_embedding}_seed{args.seed}"
        )
    return (
        f"Teleport-{args.tele_method}-lr{args.lr}{adam_lr_suffix}_batch{args.batch_size}_embeded{args.position_embedding}_opt{args.opt}"
        f"att{args.tele_att}_mlp{args.tele_mlp}_"
        f"tele-opt{args.tele_opt}_tele-high{args.tele_high}_tele-low{args.tele_low}_tele-cons{args.tele_cons}_tele-sign{args.tele_sign}_"
        f"tele-layer{args.tele_layer}_tele-batch{args.tele_batch}_tele-epoch{args.tele_epoch_array}_cands{args.tele_candidates}_seed{args.seed}"
    )


def maybe_teleport(args):
    return args.tele_method != "none" or args.n_teleport > 0



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

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    if args.tele_method == "naive" and args.n_teleport <= 0:
        args.n_teleport = max(args.tele_candidates, 2)

    torch.manual_seed(args.seed)
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")

    run_name = build_run_name(args)
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(args, device)
    print("\n========== NUMERICAL CONFIGURATION ==========")
    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
    if device.type == "cuda":
        print("GPU capability:", torch.cuda.get_device_capability(device))
    print("Model dtype:", next(model.parameters()).dtype)
    print("Attention implementation:", getattr(model.config, "_attn_implementation", "unknown"))
    print("TF32 matmul:", torch.backends.cuda.matmul.allow_tf32)
    print("TF32 cuDNN:", torch.backends.cudnn.allow_tf32)
    print("Float32 matmul precision:", torch.get_float32_matmul_precision())
    print("==============================================\n")

    train_loader, val_loader = get_dataset(args)
    optimizer = build_optimizer(model, args)

    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
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
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            processed_batches = 0

            for batch_index, (images, labels) in enumerate(
                tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Training")
            ):
                if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
                    break

                images = images.to(device)
                labels = labels.to(device)

                teleport_event = (
                    maybe_teleport(args)
                    and tele_scheduler is not None
                    and tele_scheduler[batch_index]
                    and epoch in args.tele_epoch_array
                )
                if teleport_event:
                    method = args.tele_method
                    if method == "none" and args.n_teleport > 0:
                        method = "naive"
                    teleport_args = argparse.Namespace(**vars(args))
                    teleport_args.tele_method = method
                    try_teleportation(
                        model=model,
                        optimizer=optimizer,
                        batch=(images, labels),
                        criterion=lambda logits, target: criterion(logits, target),
                        args=teleport_args,
                        step_index=global_step,
                    )

                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                running_loss += float(loss.detach().item())
                processed_batches += 1
                global_step += 1

            avg_train_loss = running_loss / max(processed_batches, 1)
            # These are the rates that were used during this epoch (before scheduler.step).
            learning_rate = float(optimizer.param_groups[0]["lr"])
            adam_learning_rate = (
                float(optimizer.param_groups[1]["lr"])
                if args.opt == "muon" and len(optimizer.param_groups) > 1
                else None
            )

            model.eval()
            correct = 0
            total = 0
            val_loss_sum = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(pixel_values=images).logits
                    loss = criterion(logits, labels)
                    val_loss_sum += float(loss.item())
                    predictions = logits.argmax(dim=1)
                    correct += int((predictions == labels).sum().item())
                    total += int(labels.size(0))

            avg_val_accuracy = correct / max(total, 1)
            avg_val_loss = val_loss_sum / max(len(val_loader), 1)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_train_loss,
                "val_acc": avg_val_accuracy,
                "lr": learning_rate,
            }
            torch.save(checkpoint, os.path.join(save_dir, "last.pt"))

            if avg_val_accuracy > best_acc:
                best_acc = avg_val_accuracy
                best_epoch = epoch
                torch.save(checkpoint, os.path.join(save_dir, "best.pt"))

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_accuracy)
            elif scheduler is not None:
                scheduler.step()

            # One W&B write per epoch: no per-batch, sample, timing, diagnostics,
            # teleportation-event, histogram, artifact, or candidate-table logging.
            log_payload = {
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "validation/loss": avg_val_loss,
                "validation/accuracy": avg_val_accuracy,
            }
            if args.opt == "muon":
                log_payload["train/muon_learning_rate"] = learning_rate
                if adam_learning_rate is not None:
                    log_payload["train/adam_learning_rate"] = adam_learning_rate
            else:
                log_payload["train/learning_rate"] = learning_rate
            run.log(log_payload, step=epoch)


if __name__ == "__main__":
    main()
