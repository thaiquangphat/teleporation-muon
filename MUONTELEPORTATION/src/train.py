import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, SGD
from tqdm import tqdm

from src.datamodule import get_dataset
from src.logger import TrainingLogger
from src.model import SimpleMLP
from src.muon import create_muon_optimizer
from src.teleport import apply_teleportation_to_real_model, try_teleportation


def list_of_ints(value: str):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Muon-aware teleportation prototype on MNIST and CIFAR-10.")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10"], required=True)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--opt", type=str, default="muon", choices=["adamw", "sgd", "muon"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--padding", type=int, default=4)

    parser.add_argument("--tele-method", type=str, default="one_step_loss", choices=["none", "naive", "grad_norm", "muon_alignment", "one_step_loss"])
    parser.add_argument("--tele-state-mode", type=str, default="keep", choices=["keep", "transport", "reset"])
    parser.add_argument("--tele-candidates", type=int, default=6)
    parser.add_argument("--tele-interval", type=int, default=1)
    parser.add_argument("--tele-start-step", type=int, default=0)
    parser.add_argument("--tele-limit", type=int, default=-1)
    parser.add_argument("--tele-low", type=float, default=0.7)
    parser.add_argument("--tele-high", type=float, default=1.5)
    parser.add_argument("--tele-accept-tol", type=float, default=1e-6)
    parser.add_argument("--warp-loss", action="store_true")
    return parser.parse_args()


def build_optimizer(model, args):
    if args.opt == "adamw":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.opt == "sgd":
        return SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return create_muon_optimizer(model, lr=args.muon_lr, weight_decay=args.weight_decay, momentum=args.muon_momentum)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_dim = 28 * 28 if args.dataset == "MNIST" else 32 * 32 * 3
    model = SimpleMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=args.num_classes).to(device)
    train_loader, val_loader = get_dataset(args)
    optimizer = build_optimizer(model, args)
    criterion = nn.CrossEntropyLoss()

    run_name = (
        f"{args.dataset}_{args.opt}_{args.tele_method}_{args.tele_state_mode}_"
        f"seed{args.seed}_lr{args.lr}_batch{args.batch_size}_cands{args.tele_candidates}_interval{args.tele_interval}"
    )
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger(
        save_dir=str(save_dir),
        run_name=run_name,
        config=vars(args),
        environment={"device": str(device), "torch_version": torch.__version__},
        dataset={"name": args.dataset, "train_samples": len(train_loader.dataset), "validation_samples": len(val_loader.dataset)},
        model={"parameter_count": sum(parameter.numel() for parameter in model.parameters())},
    )

    best_acc = 0.0
    best_epoch = 0
    step_index = 0
    with logger:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
                images, labels = images.to(device), labels.to(device)

                if args.tele_method != "none" and step_index >= args.tele_start_step and (args.tele_limit < 0 or step_index < args.tele_limit):
                    if (step_index % args.tele_interval) == 0:
                        accepted, info = try_teleportation(model, optimizer, (images, labels), criterion, args, step_index)
                        if accepted:
                            apply_teleportation_to_real_model(
                                model,
                                torch.tensor(info["candidate"], device=device),
                                args.tele_state_mode,
                                optimizer,
                            )
                            logger.data.setdefault("telemetry", []).append(info)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                step_index += 1

            model.eval()
            total_correct = 0
            total = 0
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_acc = total_correct / max(total, 1)
            avg_loss = running_loss / max(len(train_loader), 1)
            print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, save_dir / "best.pt")

            logger.log_epoch({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": val_loss / max(len(val_loader), 1),
                "val_acc": val_acc,
                "best_val_acc": best_acc,
                "best_epoch": best_epoch,
            }, best_acc, best_epoch)

    config_path = save_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Completed run: {save_dir}")


if __name__ == "__main__":
    main()
