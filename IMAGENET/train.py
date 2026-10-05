import argparse
import os
import torch
import wandb
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import ViTConfig, ViTForImageClassification
from muon import create_muon_optimizer
from model import ViTRoPEForImageClassification
from datasets import build_dataset
from teleport import try_teleportation, generate_tele_scheduler, calculate_grad_L2
from utils import set_seed, save_checkpoint, remove_old_checkpoints

# ---------- Dataset Loader ----------
def data_loader(args):
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    # dataset_train, args.nb_classes = build_dataset(is_train=False, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )
    return data_loader_train, data_loader_val

def main(args: argparse.Namespace):
    # Init wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        id=args.wandb_id,
        name=f"lr{args.lr}-warmup{args.warmup_lr}-{args.opt}-teleport{args.n_teleport}-batch{args.tele_batch}-att{args.tele_att}-mlp{args.tele_mlp}"
            f"-{args.tele_opt}-high{args.tele_high}-low{args.tele_low}-start{args.tele_start}-limit{args.tele_limit}-sign{args.tele_sign}",
        save_code=True,
    )
    save_path = os.path.join(args.save_dir,wandb.run.name)
    wandb.config.update(vars(args))
    set_seed(args.seed)
    save_path = os.path.join(args.save_dir, wandb.run.name)
    os.makedirs(save_path, exist_ok=True)
    # === Model and Data  ===
    config = ViTConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layer,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        image_size=args.input_size,
        patch_size=args.patch_size,
        num_channels=args.num_channels,
        num_labels=args.num_classes,
        qkv_bias=True,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    config.save_pretrained(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.position_embedding == "rope":
        model = ViTRoPEForImageClassification(config).to(device)
    else:
        model = ViTForImageClassification(config).to(device)
    train_loader, val_loader = data_loader(args)
    # Optimizer and Scheduler
    start_factor = args.warmup_lr / args.lr
    if(args.opt == "sgd"):
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif(args.opt == "adamw"):
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif(args.opt == "muon"):
        optimizer = create_muon_optimizer(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    start_factor = args.warmup_lr / args.lr
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.eta_min)
    scheduler = SequentialLR(optimizer,schedulers=[warmup_scheduler, cosine_scheduler],milestones=[args.warmup_epochs])
    curr_epoch, best_acc = 1, 0.0
    if(args.restore_path):
        checkpoint = torch.load(args.restore_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        curr_epoch = checkpoint['epoch'] + 1  # resume after saved epoch
        best_acc = checkpoint['val_acc']
        current_lr = checkpoint['learning_rate']
    # Teleport Setting
    tele_scheduler = generate_tele_scheduler(tele_batch=args.tele_batch, number_of_batch=len(train_loader),tele_opt=args.tele_opt,tele_cons=args.tele_cons)    
    # Training Setting
    criterion = nn.CrossEntropyLoss()
    global_step = (curr_epoch-1)*len(train_loader)
    print("Starting training...")
    for epoch in range(curr_epoch, args.epochs+1):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        progress_bar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Training")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            if(args.n_teleport and tele_scheduler[batch_idx] and (epoch)%args.tele_epoch == 0  and args.tele_start<= epoch <= args.tele_limit):
                try_teleportation(
                    vit_model=model, criterion=lambda logits, target: criterion(logits, target), samples=images,targets=labels,args = args,
                    high = args.tele_high, low = args.tele_low, sign = args.tele_sign
                )                
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            grad_L2 = calculate_grad_L2(model)
            optimizer.step()
            preds = outputs.argmax(dim=1)
            train_loss_step = loss.item()
            train_acc_step  = (preds == labels).sum().item()*100/ labels.size(0)
            train_loss += train_loss_step
            train_acc  += train_acc_step
            if global_step % args.logs_frequency == 0 and global_step > 0:
                wandb.log({"train_loss_step": train_loss_step, "train_acc_step": train_acc_step,"grad_L2_step": grad_L2}, step=global_step)
            # Update postfix for tqdm
            progress_bar.set_postfix({
                "loss": f"{train_loss_step:.4f}", "acc":f"{train_acc_step:.4f}", "grad_L2": f"{grad_L2:.4f}","lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            global_step += 1
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
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
        print(f"Epoch {epoch}, Train Loss:{avg_train_loss:.4f}, Train Acc:{avg_train_acc:.2f}, Val Loss:{avg_val_loss:.4f}, Val Acc:{avg_val_acc*100 :.2f}%, LR:{current_lr:.4f}")
        wandb.log({
            "train_loss": avg_train_loss, "train_acc":avg_train_acc, "val_acc": avg_val_acc,"val_loss": avg_val_loss,
            "learning_rate": current_lr, "epoch": epoch}, step=global_step
        )
        # Step the LR scheduler
        scheduler.step()
        # Save last checkpoint
        remove_old_checkpoints(save_path, "last_")
        last_path = os.path.join(save_path, f"last_{epoch}.pt")
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_acc, scheduler.get_last_lr()[0])
        # Save best checkpoint if improved
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            remove_old_checkpoints(save_path, "best_")
            best_path = os.path.join(save_path, f"best_{epoch}.pt")
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_acc, scheduler.get_last_lr()[0])
            print(f"Best model saved at epoch {epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT on CIFAR-10")
    # --- Data Config ---
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--color-jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--train-interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    # Model hyperparameters
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layer", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--position-embedding", type=str, default=None, choices=["rope", "learnable"])
    # Optimizer settings
    parser.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adamw", "muon"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["step", "cosine", "plateau","cosine_warm"])
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--eta-min', type=float, default=1e-5, metavar='LR',help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',help='epochs to warmup LR, if scheduler supports')
    # Save and Logging
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--restore-path", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity for logging")
    parser.add_argument("--wandb-id", type=str, default=None, help="wandb id for logging")
    parser.add_argument("--logs-frequency", type=int, default=50, help="Log training loss every N steps")
    # Teleport Hyperparams
    parser.add_argument('--n-teleport', type=int, default=2, help='number of teleport')
    parser.add_argument('--tele-epoch', type=int, default=2, help='Distance of teleports')
    parser.add_argument('--tele-batch', type=int, default=32, help='Number of batch per Tele')
    parser.add_argument('--tele-start', type=int, default=0, help='Limit of tele epoch')
    parser.add_argument('--tele-limit', type=int, default=100, help='Limit of tele epoch')
    parser.add_argument('--tele-att', type=int, default=1, help='Teleport Attention')
    parser.add_argument('--tele-mlp', type=int, default=0, help='Teleport MLP')
    parser.add_argument('--tele-opt', type=int, default=1, help='Teleport Option')
    parser.add_argument('--tele-high', type=float, default=1.1, help='Teleport Max Entries ')
    parser.add_argument('--tele-low', type=float, default=0.9, help='Teleport Min Entries')
    parser.add_argument('--tele-cons', type=int, default=16, help='Teleport Consecutive')
    parser.add_argument('--tele-sign', type=int, default=0, help='Sign')
    parser.add_argument('--tele-layer', type=str, default="all", choices=["all", "first", "last"])
    args = parser.parse_args()
    main(args)
