import argparse
import os
import time
import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification
from muon import create_muon_optimizer
from datamodule import get_dataset
from teleport import try_teleportation, generate_tele_scheduler, calculate_grad_L2
from model import ViTRoPEForImageClassification
from training_logger import TrainingLogger

def list_of_ints(arg):
    return [int(x) for x in arg.split(',')]

def main():
    parser = argparse.ArgumentParser(description="Train ViT Teleportation")

    # Dataset and Dataloader
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str,choices=["CIFAR10","MNIST"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    # Device
    parser.add_argument("--devices", type=str, default="cuda:0")
    # Image Augmentation
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--num_channels", type=int, default=1)
    # Model hyperparameters
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-hidden-layer", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--position-embedding", type=str, default="learnable", choices=["rope", "learnable"])
    # Optimizer settings
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--opt", type=str, default="adamw", choices=["adamw", "sgd", "muon"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["step", "cosine", "plateau"])
    # Teleport Hyperparams
    parser.add_argument('--n-teleport', type=int, default=0, help='number of teleport')
    # parser.add_argument('--tele-epoch', type=int, default=2, help='Epoch of teleports')
    parser.add_argument('--tele-batch', type=int, default=32, help='Number of batch per Tele')
    # parser.add_argument('--tele-limit', type=int, default=10, help='Limit of tele epoch')
    parser.add_argument('--tele-epoch-array', type=list_of_ints, default=[1], help='Epoch of teleports')
    parser.add_argument('--tele-att', type=int, default=1, help='Teleport Attention')
    parser.add_argument('--tele-mlp', type=int, default=0, help='Teleport MLP')
    parser.add_argument('--tele-opt', type=int, default=1, help='Teleport Option')
    parser.add_argument('--tele-high', type=float, default=1.1, help='Teleport Max Entries ')
    parser.add_argument('--tele-low', type=float, default=0.9, help='Teleport Min Entries')
    parser.add_argument('--tele-cons', type=int, default=16, help='Consecutive')
    parser.add_argument('--tele-sign', type=int, default=0, help='Sign')
    parser.add_argument('--tele-layer', type=str, default="all", choices=["all", "first", "last"])

    args = parser.parse_args()

    # Set seed and device
    torch.manual_seed(args.seed)
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")

    if args.n_teleport > 0:
        run_name = (
            f"Teleport-lr{args.lr}_batch{args.batch_size}_embeded{args.position_embedding}_opt{args.opt}"
            f"att{args.tele_att}_mlp{args.tele_mlp}_"
            f"tele-opt{args.tele_opt}_tele-high{args.tele_high}_tele-low{args.tele_low}_tele-cons{args.tele_cons}_tele-sign{args.tele_sign}_"
            f"tele-layer{args.tele_layer}_tele-batch{args.tele_batch}_tele-epoch{args.tele_epoch_array}_seed{args.seed}"
        )
    else:
        run_name = (
            f"NoTeleport-lr{args.lr}_batch{args.batch_size}_{args.opt}_embed{args.position_embedding}_seed{args.seed}"
        )

    # Create save directory
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print("Save directory: ", save_dir)

    # === Create ViT Configuration ===
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
        model = ViTRoPEForImageClassification(config).to(device)
    else:
        model = ViTForImageClassification(config).to(device)
    train_loader, val_loader = get_dataset(args)

    if args.opt == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.opt == "muon":
        optimizer = create_muon_optimizer(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )

    # Scheduler: Cosine
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    else:
        scheduler = None

    if (args.n_teleport > 0): 
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
    logger = TrainingLogger(
        save_dir=save_dir,
        run_name=run_name,
        config=vars(args),
        environment={
            "device": str(device),
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        dataset={
            "name": args.dataset,
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset),
            "train_batches_per_epoch": len(train_loader),
            "validation_batches_per_epoch": len(val_loader),
        },
        model={
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
    )

    with logger:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Training")):
                images, labels = images.to(device), labels.to(device)

                # === Optional Teleportation ===
                if (args.n_teleport > 0) and tele_scheduler[i] and epoch in args.tele_epoch_array:
                    try_teleportation(
                        vit_model=model,
                        criterion=lambda logits, target: criterion(logits, target),
                        samples=images,
                        targets=labels,
                        args=args,
                        high=args.tele_high,
                        low=args.tele_low,
                        sign=args.tele_sign,
                    )

                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
                global_step += 1

            train_time_seconds = time.perf_counter() - epoch_started
            avg_loss = running_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

            # === Validation ===
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
            validation_time_seconds = time.perf_counter() - validation_started
            avg_val_acc = correct / total
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch}, Val Acc: {avg_val_acc * 100:.2f}%, Val Loss: {avg_val_loss:.4f}")

            # Save last checkpoint
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

            # Save best checkpoint if improved
            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                best_epoch = epoch
                best_ckpt_path = os.path.join(save_dir, "best.pt")
                torch.save(checkpoint, best_ckpt_path)

            # Step the LR scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_acc)
            elif scheduler is not None:
                scheduler.step()

            epoch_time_seconds = time.perf_counter() - epoch_started
            logger.log_epoch(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": avg_val_loss,
                    "val_acc": avg_val_acc,
                    "lr": current_lr,
                    "next_lr": optimizer.param_groups[0]["lr"],
                    "train_time_seconds": round(train_time_seconds, 6),
                    "validation_time_seconds": round(validation_time_seconds, 6),
                    "epoch_time_seconds": round(epoch_time_seconds, 6),
                    "global_step": global_step,
                },
                best_val_acc=best_acc,
                best_epoch=best_epoch,
            )
    print(f"Training log: {logger.log_path}")

if __name__ == "__main__":
    main()