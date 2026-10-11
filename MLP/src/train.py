import argparse
import os
import random
import time

import torch
from torch import nn
from torch.optim import AdamW, SGD
from tqdm import tqdm

from datamodule import get_dataset
from logger import TrainingLogger
from model import SimpleMLP
from muon import SingleDeviceMuonWithAuxAdam
from teleport import generate_tele_scheduler, try_teleportation


def list_of_ints(arg):
    return [int(x) for x in arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train a simple MLP with optional teleportation")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--hidden-dims", type=list_of_ints, default=[256, 128])
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--opt", type=str, default="adamw", choices=["adamw", "sgd", "muon"])
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)

    parser.add_argument("--n-teleport", type=int, default=0)
    parser.add_argument("--tele-batch", type=int, default=32)
    parser.add_argument("--tele-opt", type=int, default=1)
    parser.add_argument("--tele-high", type=float, default=1.1)
    parser.add_argument("--tele-low", type=float, default=0.9)
    parser.add_argument("--tele-cons", type=int, default=16)
    parser.add_argument("--tele-sign", type=int, default=0)
    parser.add_argument("--tele-layer", type=str, default="all", choices=["all", "first", "last"])
    parser.add_argument("--tele-mlp", type=int, default=1)
    parser.add_argument("--tele-epoch-array", type=list_of_ints, default=[1])

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset == "MNIST":
        input_dim = 28 * 28
    else:
        input_dim = 32 * 32 * 3

    run_name = (
        f"{args.dataset}-MLP-lr{args.lr}_batch{args.batch_size}_hd{args.hidden_dims}_opt{args.opt}"
        f"_tele{args.n_teleport}_telebatch{args.tele_batch}_telehigh{args.tele_high}"
        f"_telelow{args.tele_low}_seed{args.seed}"
        f"_muonlr{args.muon_lr if args.opt == 'muon' else 'na'}"
    )
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")

    model = SimpleMLP(input_dim=input_dim, hidden_dims=args.hidden_dims, num_classes=args.num_classes).to(device)
    train_loader, val_loader = get_dataset(args)

    if args.opt == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        linear_layers = [module for module in model.net.modules() if isinstance(module, nn.Linear)]
        muon_parameters = [parameter for layer in linear_layers[:-1] for parameter in (layer.weight,)]
        muon_parameter_ids = {id(parameter) for parameter in muon_parameters}
        adam_parameters = [
            parameter for parameter in model.parameters()
            if id(parameter) not in muon_parameter_ids
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": muon_parameters,
                    "use_muon": True,
                    "lr": args.muon_lr,
                    "momentum": args.muon_momentum,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": adam_parameters,
                    "use_muon": False,
                    "lr": args.lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-10,
                    "weight_decay": args.weight_decay,
                },
            ]
        )

    criterion = nn.CrossEntropyLoss()
    tele_scheduler = generate_tele_scheduler(
        tele_batch=args.tele_batch,
        number_of_batch=len(train_loader),
        tele_opt=args.tele_opt,
        tele_cons=args.tele_cons,
    ) if args.n_teleport > 0 else None

    best_acc = 0.0
    best_epoch = 0
    logger = TrainingLogger(
        save_dir=save_dir,
        run_name=run_name,
        config=vars(args),
        environment={"device": str(device), "torch_version": torch.__version__},
        dataset={
            "name": args.dataset,
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset),
            "train_batches_per_epoch": len(train_loader),
            "validation_batches_per_epoch": len(val_loader),
        },
        model={"parameter_count": sum(p.numel() for p in model.parameters())},
    )

    with logger:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            running_loss = 0.0
            training_started = time.perf_counter()
            for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}]")):
                images, labels = images.to(device), labels.to(device)

                if args.tele_mlp and args.n_teleport > 0 and tele_scheduler is not None and tele_scheduler[i] and epoch in args.tele_epoch_array:
                    try_teleportation(
                        model=model,
                        samples=images,
                        targets=labels,
                        criterion=criterion,
                        args=args,
                        high=args.tele_high,
                        low=args.tele_low,
                        sign=args.tele_sign,
                    )

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_time_seconds = time.perf_counter() - training_started
            avg_loss = running_loss / len(train_loader)
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            validation_started = time.perf_counter()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            validation_time_seconds = time.perf_counter() - validation_started
            avg_val_acc = correct / total
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_acc={avg_val_acc:.4f}, val_loss={avg_val_loss:.4f}")

            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                best_epoch = epoch
                checkpoint_path = os.path.join(save_dir, "best.pt")
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_acc": avg_val_acc}, checkpoint_path)

            logger.log_epoch(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": avg_val_loss,
                    "val_acc": avg_val_acc,
                    "timestamp": time.time(),
                    "train_time_seconds": round(train_time_seconds, 6),
                    "validation_time_seconds": round(validation_time_seconds, 6),
                    "epoch_time_seconds": round(time.perf_counter() - epoch_started, 6),
                },
                best_val_acc=best_acc,
                best_epoch=best_epoch,
            )

    print(f"Training log: {logger.log_path}")


if __name__ == "__main__":
    main()
