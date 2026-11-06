#!/usr/bin/env python3
"""Fine-tune pretrained GPT-2 on WikiText with regular Muon or CSD-Muon teleportation."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import wandb
import yaml
from tqdm import tqdm

from datamodule import get_wikitext_dataloaders
from model import GPT2LMForCausalLM
from teleport import create_lm_muon_optimizer, generate_tele_scheduler
from csd_nesterov import try_teleportation


def list_of_ints(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-2 CSD-Muon on WikiText")
    parser.add_argument("--data-dir", type=str, default="./data/wt103")
    parser.add_argument("--dataset-source", type=str, default="local", choices=["local", "huggingface"])
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--position-embedding", type=str, default="ape", choices=["ape", "rope"])
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--force-retokenize", action="store_true")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--devices", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./checkpoints-wikitext")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--amp-dtype", type=str, default="none", choices=["none", "bfloat16"])

    parser.add_argument("--opt", type=str, default="muon", choices=["muon", "adamw"])
    parser.add_argument("--lr", type=float, default=0.002, help="Muon LR; used as AdamW LR when --opt=adamw")
    parser.add_argument("--adam-lr", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="Use Nesterov-corrected Muon updates (enabled for the CSD-Nesterov experiment only).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["none", "cosine", "step", "plateau"])
    parser.add_argument("--grad-clip", type=float, default=0.0)

    parser.add_argument("--tele-method", type=str, default="none", choices=["none", "naive", "csd_l2_nesterov", "muon_alignment"])
    parser.add_argument("--tele-epoch-array", type=list_of_ints, default=[1])
    parser.add_argument("--tele-att", type=int, default=1)
    parser.add_argument("--tele-mlp", type=int, default=0)
    parser.add_argument("--tele-layer", type=str, default="all", choices=["all", "first", "last"])
    parser.add_argument("--tele-candidates", type=int, default=6)
    parser.add_argument("--n-teleport", type=int, default=6, help="Total naive candidates, including identity")
    parser.add_argument("--tele-high", type=float, default=1.65)
    parser.add_argument("--tele-low", type=float, default=0.35)
    parser.add_argument("--tele-sign", type=int, default=0)
    parser.add_argument("--tele-batch", type=int, default=4)
    parser.add_argument("--tele-cons", type=int, default=4)
    parser.add_argument("--tele-opt", type=int, default=1, choices=[0, 1])

    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wandb_settings(args: argparse.Namespace) -> tuple[str, Optional[str], str]:
    config_path = os.environ.get("WANDB_CONFIG_PATH", "./wandb.yaml")
    file_config = {}
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as config_file:
            file_config = yaml.safe_load(config_file) or {}
        if not isinstance(file_config, dict):
            raise ValueError(f"W&B config must be a YAML mapping: {config_path}")

    mode = args.wandb_mode or os.environ.get("WANDB_MODE") or file_config.get("mode") or "online"
    project = args.wandb_project or os.environ.get("WANDB_PROJECT") or file_config.get("project")
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or file_config.get("entity") or None
    api_key = os.environ.get("WANDB_API_KEY") or file_config.get("api_key")
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("W&B mode must be online, offline, or disabled")
    if not project and mode == "disabled":
        project = "muon-teleportation-wikitext-gpt2"
    if not project:
        raise ValueError("Set project in wandb.yaml, WANDB_PROJECT, or --wandb-project")
    if mode == "online" and not api_key:
        raise ValueError("Set api_key in wandb.yaml or WANDB_API_KEY for online W&B")
    if api_key:
        os.environ["WANDB_API_KEY"] = str(api_key)
    return str(project), str(entity) if entity else None, str(mode)


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.opt == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    return create_lm_muon_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        adam_lr=args.adam_lr,
        nesterov=args.nesterov,
    )


def build_run_name(args: argparse.Namespace) -> str:
    model_tag = args.model_name.replace("/", "-")
    if args.tele_method == "naive":
        teleport_tag = (
            f"-ntele{args.n_teleport}-att{args.tele_att}-mlp{args.tele_mlp}"
            f"-layer{args.tele_layer}-epochs{'_'.join(map(str, args.tele_epoch_array))}"
        )
    elif args.tele_method in {"csd_l2_nesterov", "muon_alignment"}:
        teleport_tag = (
            f"-cands{args.tele_candidates}-att{args.tele_att}-mlp{args.tele_mlp}"
            f"-layer{args.tele_layer}-epochs{'_'.join(map(str, args.tele_epoch_array))}"
        )
    else:
        teleport_tag = ""
    return (
        f"WikiText-{args.dataset_config}-{model_tag}-{args.position_embedding}-"
        f"{args.opt}-lr{args.lr}-adam{args.adam_lr}-bs{args.batch_size}-"
        f"block{args.block_size}-{args.tele_method}-nag{int(args.nesterov)}{teleport_tag}-seed{args.seed}"
    )


def autocast_context(args: argparse.Namespace, device: torch.device):
    if device.type == "cuda" and args.amp_dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support bfloat16 autocast")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    args: argparse.Namespace,
    desc: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    processed_batches = 0
    with torch.no_grad():
        for batch_index, (input_ids, targets) in enumerate(tqdm(loader, desc=desc, leave=False)):
            if args.max_eval_batches > 0 and batch_index >= args.max_eval_batches:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(args, device):
                logits = model(input_ids=input_ids).logits
                loss = compute_loss(logits, targets, criterion)
            count = int(targets.numel())
            total_loss += float(loss.item()) * count
            total_tokens += count
            processed_batches += 1
    if total_tokens == 0:
        raise RuntimeError("Evaluation processed zero tokens")
    mean_loss = total_loss / total_tokens
    perplexity = math.exp(min(mean_loss, 20.0))
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "tokens": float(total_tokens),
        "batches": float(processed_batches),
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if args.tele_method in {"naive", "csd_l2_nesterov", "muon_alignment"} and args.opt != "muon":
        raise ValueError("LM teleportation with optimizer-state transport requires --opt muon")
    if args.tele_method in {"naive", "csd_l2_nesterov", "muon_alignment"} and not (args.tele_att or args.tele_mlp):
        raise ValueError("Enable --tele-att 1 and/or --tele-mlp 1 for teleportation")
    if args.tele_method == "naive" and args.n_teleport < 2:
        raise ValueError("Naive teleportation requires --n-teleport >= 2")
    if args.tele_method in {"csd_l2_nesterov", "muon_alignment"} and args.tele_candidates < 2:
        raise ValueError("Nesterov CSD-L2 requires --tele-candidates >= 2")
    if args.tele_method in {"csd_l2_nesterov", "muon_alignment"} and not args.nesterov:
        raise ValueError(
            "The Nesterov CSD-L2 experiment requires --nesterov so the actual Muon update "
            "matches the direction used by CSD scoring."
        )

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")

    run_name = build_run_name(args)
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    setup_started = time.perf_counter()
    data_started = time.perf_counter()
    train_loader, val_loader, test_loader, tokenizer, data_sizes = get_wikitext_dataloaders(
        dataset_config=args.dataset_config,
        model_name=args.model_name,
        block_size=args.block_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        force_retokenize=args.force_retokenize,
        dataset_source=args.dataset_source,
    )
    data_loading_seconds = time.perf_counter() - data_started

    model_started = time.perf_counter()
    model, load_report = GPT2LMForCausalLM.from_pretrained(
        model_name=args.model_name,
        position_embedding=args.position_embedding,
        cache_dir=args.cache_dir,
    )
    if args.block_size > model.config.n_positions:
        raise ValueError(
            f"block_size={args.block_size} exceeds GPT-2 context length "
            f"n_positions={model.config.n_positions}"
        )
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match GPT-2 embedding "
            f"vocabulary ({model.config.vocab_size}). Avoid adding tokenizer tokens."
        )
    model = model.to(device)
    optimizer = build_optimizer(model, args)
    model_setup_seconds = time.perf_counter() - model_started

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1
        )
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    project, entity, mode = load_wandb_settings(args)
    tokenizer.save_pretrained(save_dir)

    print("\n========== GPT-2 WIKITEXT CONFIGURATION ==========")
    print("Device:", device)
    print("Position embedding:", args.position_embedding)
    print("Pretrained checkpoint:", args.model_name)
    print("Checkpoint load report:", load_report)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("GPT-2 hidden/layers/heads:", model.config.n_embd, model.config.n_layer, model.config.n_head)
    print("Train/validation/test blocks:", data_sizes)
    print("=================================================\n")

    with wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=vars(args),
        dir=save_dir,
        mode=mode,
        job_type="language-model-finetuning",
    ) as run:
        run_started = time.perf_counter()
        run.config.update(
            {
                "environment_device": str(device),
                "environment_gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "environment_torch_version": torch.__version__,
                "environment_cuda_version": torch.version.cuda or "none",
                "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "model_vocab_size": model.config.vocab_size,
                "model_context_length": model.config.n_positions,
                "model_hidden_size": model.config.n_embd,
                "model_layers": model.config.n_layer,
                "model_heads": model.config.n_head,
                "dataset_sizes": data_sizes,
                "pretrained_load_report": load_report,
            },
            allow_val_change=True,
        )
        run.log(
            {
                "timing/data_loading_seconds": data_loading_seconds,
                "timing/model_setup_seconds": model_setup_seconds,
                "timing/initialization_seconds": time.perf_counter() - setup_started,
            }
        )

        attempted_teleports = 0
        accepted_teleports = 0
        global_step = 0
        best_val_loss = float("inf")
        best_epoch = 0
        total_train_seconds = 0.0
        total_validation_seconds = 0.0
        telemetry_table = wandb.Table(
            columns=[
                "epoch", "batch", "global_step", "method", "accepted", "reason",
                "search_time_seconds", "selection_metric", "baseline_score",
                "selected_score", "score_gain", "gradient_l2_gain", "csd_l2_gain",
                "selected_candidate_index", "num_candidates_evaluated",
            ]
        )
        candidate_table = wandb.Table(
            columns=[
                "epoch", "batch", "global_step", "candidate_index", "is_identity",
                "selection_metric", "selection_score", "gradient_l2",
                "csd_l2_score", "csd_l2_squared_score", "csd_l2_rank",
                "scale_min", "scale_max", "transform_type",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            synchronize_device(device)
            epoch_started = time.perf_counter()
            model.train()
            tele_scheduler = None
            if args.tele_method in {"naive", "csd_l2_nesterov", "muon_alignment"} and epoch in args.tele_epoch_array:
                tele_scheduler = generate_tele_scheduler(
                    tele_batch=args.tele_batch,
                    number_of_batch=len(train_loader),
                    tele_opt=args.tele_opt,
                    tele_cons=args.tele_cons,
                )

            total_train_loss = 0.0
            total_train_tokens = 0
            processed_batches = 0
            for batch_index, (input_ids, targets) in enumerate(
                tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Training")
            ):
                if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
                    break
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if tele_scheduler is not None and batch_index < len(tele_scheduler) and tele_scheduler[batch_index]:
                    accepted, info = try_teleportation(
                        model=model,
                        optimizer=optimizer,
                        batch=(input_ids, targets),
                        criterion=criterion,
                        args=args,
                        step_index=global_step,
                    )
                    attempted_teleports += 1
                    accepted_teleports += int(accepted)
                    telemetry_table.add_data(
                        epoch,
                        batch_index,
                        global_step,
                        info.get("method", args.tele_method),
                        accepted,
                        info.get("reason", ""),
                        info.get("search_time_seconds", 0.0),
                        info.get("selection_metric"),
                        info.get("baseline_score"),
                        info.get("selected_score"),
                        info.get("score_gain"),
                        info.get("gradient_l2_gain"),
                        info.get("csd_l2_gain"),
                        info.get("selected_candidate_index"),
                        info.get("num_candidates_evaluated"),
                    )
                    for candidate in info.get("results", []):
                        candidate_table.add_data(
                            epoch,
                            batch_index,
                            global_step,
                            candidate.get("candidate_index"),
                            candidate.get("identity"),
                            info.get("selection_metric"),
                            candidate.get("selection_score", candidate.get("csd_l2_score")),
                            candidate.get("gradient_l2"),
                            candidate.get("csd_l2_score"),
                            candidate.get("csd_l2_squared_score"),
                            candidate.get("csd_l2_rank"),
                            candidate.get("scale_min"),
                            candidate.get("scale_max"),
                            candidate.get("transform_type"),
                        )
                    run.log(
                        {
                            "teleport/attempted": attempted_teleports,
                            "teleport/accepted": accepted_teleports,
                            "teleport/acceptance_rate": accepted_teleports / attempted_teleports,
                            "teleport/search_time_seconds": info.get("search_time_seconds", 0.0),
                            "teleport/selection_score": info.get(
                                "selected_score", info.get("selected_csd_l2_score")
                            ),
                            "teleport/score_gain": info.get(
                                "score_gain", info.get("csd_l2_gain")
                            ),
                            "teleport/gradient_l2_gain": info.get("gradient_l2_gain"),
                            "teleport/selected_gradient_l2": info.get("selected_gradient_l2"),
                            "teleport/csd_l2_gain": info.get("csd_l2_gain"),
                            "teleport/selected_csd_l2_score": info.get("selected_csd_l2_score"),
                            "teleport/selected_candidate_index": info.get("selected_candidate_index"),
                            "teleport/accepted_event": int(accepted),
                        },
                        step=global_step,
                    )

                optimizer.zero_grad(set_to_none=True)
                with autocast_context(args, device):
                    logits = model(input_ids=input_ids).logits
                    loss = compute_loss(logits, targets, criterion)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                token_count = int(targets.numel())
                total_train_loss += float(loss.item()) * token_count
                total_train_tokens += token_count
                processed_batches += 1
                global_step += 1
                run.log(
                    {
                        "train/batch_loss": float(loss.item()),
                        "train/batch_perplexity": math.exp(min(float(loss.item()), 20.0)),
                        "train/tokens": token_count,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/muon_learning_rate": optimizer.param_groups[0]["lr"],
                        **(
                            {"train/adam_learning_rate": optimizer.param_groups[1]["lr"]}
                            if args.opt == "muon" else {}
                        ),
                    },
                    step=global_step,
                )

            if processed_batches == 0:
                raise RuntimeError("The training loader produced no batches")
            synchronize_device(device)
            train_seconds = time.perf_counter() - epoch_started
            total_train_seconds += train_seconds
            train_loss = total_train_loss / max(total_train_tokens, 1)

            validation_started = time.perf_counter()
            val_metrics = evaluate(
                model, val_loader, device, criterion, args,
                desc=f"[Epoch {epoch}/{args.epochs}] Validation",
            )
            synchronize_device(device)
            validation_seconds = time.perf_counter() - validation_started
            total_validation_seconds += validation_seconds

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "validation_loss": val_metrics["loss"],
                "validation_perplexity": val_metrics["perplexity"],
                "model_config": model.config_dict(),
                "position_embedding": args.position_embedding,
                "model_name": args.model_name,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(save_dir, "last.pt"))
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                torch.save(checkpoint, os.path.join(save_dir, "best.pt"))

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()

            run.log(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train/epoch_loss": train_loss,
                    "train/epoch_perplexity": math.exp(min(train_loss, 20.0)),
                    "train/tokens": total_train_tokens,
                    "train/batches": processed_batches,
                    "train/tokens_per_second": total_train_tokens / max(train_seconds, 1e-9),
                    "validation/loss": val_metrics["loss"],
                    "validation/perplexity": val_metrics["perplexity"],
                    "validation/tokens": val_metrics["tokens"],
                    "best/validation_loss": best_val_loss,
                    "best/validation_perplexity": math.exp(min(best_val_loss, 20.0)),
                    "best/epoch": best_epoch,
                    "timing/train_seconds": train_seconds,
                    "timing/validation_seconds": validation_seconds,
                    "timing/epoch_seconds": time.perf_counter() - epoch_started,
                    "teleport/attempted": attempted_teleports,
                    "teleport/accepted": accepted_teleports,
                    "teleport/acceptance_rate": accepted_teleports / attempted_teleports if attempted_teleports else 0.0,
                    "system/gpu_peak_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                        if device.type == "cuda" else 0.0
                    ),
                },
                step=global_step,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        best_checkpoint_path = os.path.join(save_dir, "best.pt")
        if os.path.isfile(best_checkpoint_path):
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(best_checkpoint["model_state_dict"])
        test_metrics = evaluate(
            model, test_loader, device, criterion, args, desc="Final test evaluation"
        )
        run.log(
            {
                "test/loss": test_metrics["loss"],
                "test/perplexity": test_metrics["perplexity"],
                "test/tokens": test_metrics["tokens"],
                "test/best_checkpoint_epoch": best_epoch,
            },
            step=global_step,
        )
        run.log({"teleport/events": telemetry_table, "teleport/candidates": candidate_table})
        run.summary["best/validation_loss"] = best_val_loss
        run.summary["best/validation_perplexity"] = math.exp(min(best_val_loss, 20.0))
        run.summary["best/epoch"] = best_epoch
        run.summary["test/perplexity"] = test_metrics["perplexity"]
        run.summary["teleport/attempted"] = attempted_teleports
        run.summary["teleport/accepted"] = accepted_teleports
        run.summary["timing/total_train_seconds"] = total_train_seconds
        run.summary["timing/total_validation_seconds"] = total_validation_seconds
        run.summary["timing/run_seconds"] = time.perf_counter() - run_started


if __name__ == "__main__":
    main()
