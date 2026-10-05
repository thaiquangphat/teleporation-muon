import argparse
import os
import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification
from datamodule import get_dataset
from teleport_zhao import teleport, generate_tele_scheduler
from model import ViTRoPEForImageClassification

def list_of_ints(arg):
    return [int(x) for x in arg.split(',')]

def calculate_grad_L2(net):
    """Calculates the L2 norm of the gradients of the network."""
    total_L2 = 0.0
    for param in net.parameters():
        if param.grad is not None:
            total_L2 += param.grad.norm(2).item() ** 2
    return total_L2 ** 0.5

def main():
    parser = argparse.ArgumentParser(description="Train ViT Teleportation with Gradient Ascent")

    # Dataset and Dataloader
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, choices=["CIFAR10","MNIST"])
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
    parser.add_argument("--position-embedding", type=str, default=None, choices=["rope", "learnable"])
    # Optimizer settings
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--opt", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["step", "cosine", "plateau"])
    # Teleport Hyperparams
    parser.add_argument('--tele-epoch-array', type=list_of_ints, default=[1], help='Epochs to perform teleportation, e.g., 1,3,5')
    parser.add_argument('--tele-batch', type=int, default=32, help='Number of batches to trigger teleportation within an epoch')
    parser.add_argument('--tele-cons', type=int, default=16, help='Consecutive batches for teleportation trigger')
    parser.add_argument('--tele-opt', type=int, default=1, help='Teleportation scheduler option (0: random, 1: consecutive)')
    parser.add_argument('--tele-att', type=int, default=1, help='Enable teleportation for attention layers')
    parser.add_argument('--tele-mlp', type=int, default=0, help='Enable teleportation for MLP layers')
    parser.add_argument('--tele-lr', type=float, default=1e-4, help='Learning rate for gradient ascent on transformation matrix')
    parser.add_argument('--tele-steps', type=int, default=10, help='Number of optimization steps for the transformation matrix')

    args = parser.parse_args()

    # Set seed and device
    torch.manual_seed(args.seed)
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")

    run_name = (
            f"GradAscent-lr{args.lr}_opt{args.opt}_batch{args.batch_size}_"
            f"att{args.tele_att}_mlp{args.tele_mlp}_embeded{args.position_embedding}_"
            f"tele-batch{args.tele_batch}_tele-cons{args.tele_cons}_tele-opt{args.tele_opt}_"
            f"tele-lr{args.tele_lr}_tele-steps{args.tele_steps}_"
            f"tele-epoch{args.tele_epoch_array}_seed{args.seed}"
        )

    # Create save directory
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

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
        output_hidden_states=True,  # Ensure the model outputs hidden states
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

    # Scheduler
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    else:
        scheduler = None
        
    tele_scheduler = generate_tele_scheduler(
        tele_batch=args.tele_batch,
        number_of_batch=len(train_loader),
        tele_opt=args.tele_opt,
        tele_cons=args.tele_cons,
    )

    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        grad = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Training")):
            images, labels = images.to(device), labels.to(device)
            
            # === Optional Teleportation ===
            if (args.tele_att or args.tele_mlp) and tele_scheduler[i] and epoch in args.tele_epoch_array:
                print(f"\n--- Starting Teleportation at Epoch {epoch}, Step {i} ---")
                teleport(
                    model=model,
                    samples=images, # Use current batch for optimization
                    targets=labels,
                    criterion=criterion,
                    args=args
                )
                # After teleportation, optimizer state is stale, so we zero_grad
                optimizer.zero_grad()

            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            grad_tmp = calculate_grad_L2(model)
            grad += grad_tmp
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            
        grad /= len(train_loader)
        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # === Validation ===
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

        avg_val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}, Val Acc: {avg_val_acc * 100:.2f}%", f"Val Loss: {avg_val_loss:.4f}")

        # Save last checkpoint
        last_ckpt_path = os.path.join(save_dir, "last.pt")
        torch.save(model.state_dict(), last_ckpt_path)

        # Save best checkpoint
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            best_ckpt_path = os.path.join(save_dir, "best.pt")
            torch.save(model.state_dict(), best_ckpt_path)

        if scheduler:
            scheduler.step()

if __name__ == "__main__":
    main()