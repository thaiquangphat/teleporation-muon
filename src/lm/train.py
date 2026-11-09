#!/usr/bin/env python3
"""Step-based GPT-2 training on WikiText with Muon/CSD teleportation."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
import warnings
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
from diagnostics import (
    GPT2ActivationRecorder,
    build_tensor_snapshot,
    collect_csd_diagnostics,
    collect_gradient_geometry,
    collect_global_gradient_metrics,
    collect_weight_geometry,
    collect_weight_tensors,
    compare_weight_snapshots,
    diagnostic_layer_indices,
    log_tensor_artifact,
    log_tensor_visuals,
    optimizer_diagnostic_metrics,
)


def list_of_ints(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-based GPT-2 CSD-Muon on WikiText")
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
    parser.add_argument(
        "--train-fraction", type=float, default=1.0,
        help="Fraction of fixed-length WikiText training blocks to use (e.g. 0.1).",
    )
    parser.add_argument("--data-seed", type=int, default=42, help="Seed for the fixed training-block subset")

    # Step-based training is the default. --epochs is retained as a compatibility
    # option for older scripts; it is converted to epochs * batches_per_epoch
    # only when --max-steps is not supplied.
    parser.add_argument("--max-steps", type=int, default=None, help="Total number of optimizer updates (default: 1000)")
    parser.add_argument("--epochs", type=int, default=None, help="Deprecated compatibility option; converted to steps")
    parser.add_argument("--log-interval", type=int, default=10, help="Log training metrics every N optimizer steps")
    parser.add_argument("--eval-interval", type=int, default=100, help="Run validation every N optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Linear learning-rate warmup steps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--devices", type=str, default="cuda:0")
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", type=str, default="./checkpoints-wikitext")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Optional cap per data pass; 0 uses the full training loader")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Optional cap per evaluation; 0 uses the full split")
    parser.add_argument("--amp-dtype", type=str, default="none", choices=["none", "bfloat16"])

    parser.add_argument("--opt", type=str, default="muon", choices=["muon", "adamw"])
    parser.add_argument("--lr", type=float, default=0.002, help="Muon LR; used as AdamW LR when --opt=adamw")
    parser.add_argument("--adam-lr", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--nesterov", action="store_true",
        help="Use Nesterov-corrected Muon updates (required for CSD-Nesterov).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["none", "cosine", "step", "plateau"])
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR for cosine/step schedules")
    parser.add_argument("--lr-step-size", type=int, default=1000, help="Step-scheduler decay interval after warmup")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="Step/plateau scheduler decay factor")
    parser.add_argument("--grad-clip", type=float, default=0.0)

    parser.add_argument("--tele-method", type=str, default="none", choices=["none", "naive", "csd_l2_nesterov", "muon_alignment"])
    # Retained to avoid breaking old launch commands. Epoch-gated teleportation
    # is not used by step-based training; use tele-start-step/tele-limit-step.
    parser.add_argument("--tele-epoch-array", type=list_of_ints, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tele-att", type=int, default=1)
    parser.add_argument("--tele-mlp", type=int, default=0)
    parser.add_argument("--tele-layer", type=str, default="all", choices=["all", "first", "last"])
    parser.add_argument("--tele-candidates", type=int, default=6)
    parser.add_argument("--n-teleport", type=int, default=6, help="Total naive candidates, including identity")
    parser.add_argument("--tele-high", type=float, default=1.65)
    parser.add_argument("--tele-low", type=float, default=0.35)
    parser.add_argument("--tele-sign", type=int, default=0)
    parser.add_argument("--tele-batch", type=int, default=4, help="Teleport attempts per teleport scheduling window")
    parser.add_argument("--tele-cons", type=int, default=4)
    parser.add_argument("--tele-opt", type=int, default=1, choices=[0, 1])
    parser.add_argument("--tele-interval", type=int, default=0, help="Teleport scheduling window in steps; 0 means the full run")
    parser.add_argument("--tele-start-step", type=int, default=1, help="First eligible optimizer step for teleportation (1-based)")
    parser.add_argument("--tele-limit-step", type=int, default=0, help="Last eligible step; 0 means no limit")

    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    # Research diagnostics. Zero disables the corresponding periodic collector.
    parser.add_argument("--diagnostics-interval", type=int, default=100,
                        help="Collect matrix/activation/gradient/optimizer diagnostics every N steps; 0 disables periodic diagnostics.")
    parser.add_argument("--diagnostics-spectrum-interval", type=int, default=500,
                        help="Run exact singular-value/rank diagnostics every N steps; 0 disables exact spectra.")
    parser.add_argument("--diagnostics-csd-interval", type=int, default=100,
                        help="Collect per-matrix CSD dual-norm contributions every N steps; 0 disables this collector.")
    parser.add_argument("--diagnostics-tensor-interval", type=int, default=500,
                        help="Save exact matrix/activation tensors as W&B artifacts every N steps; 0 disables periodic artifacts.")
    parser.add_argument("--diagnostics-artifact-on-teleport", action=argparse.BooleanOptionalAction, default=True,
                        help="Save before/after tensors on accepted teleportations.")
    parser.add_argument("--diagnostics-snapshot-layers", type=int, default=1,
                        help="Number of representative transformer layers (middle by default; first/middle/last when >1) in tensor artifacts and exact spectra.")
    parser.add_argument("--diagnostics-function-check", action=argparse.BooleanOptionalAction, default=True,
                        help="Check sampled logits before/after each attempted teleportation in eval mode.")
    parser.add_argument("--diagnostics-candidate-rollout", action=argparse.BooleanOptionalAction, default=False,
                        help="Expensive: take one virtual Muon update for every CSD candidate and compare post-step loss.")
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
            f"-layer{args.tele_layer}-teleint{args.tele_interval}-telebatch{args.tele_batch}"
            f"-telecons{args.tele_cons}-start{args.tele_start_step}-limit{args.tele_limit_step}"
        )
    elif args.tele_method in {"csd_l2_nesterov", "muon_alignment"}:
        teleport_tag = (
            f"-cands{args.tele_candidates}-att{args.tele_att}-mlp{args.tele_mlp}"
            f"-layer{args.tele_layer}-teleint{args.tele_interval}-telebatch{args.tele_batch}"
            f"-telecons{args.tele_cons}-start{args.tele_start_step}-limit{args.tele_limit_step}"
        )
    else:
        teleport_tag = ""
    return (
        f"WikiText-{args.dataset_config}-{model_tag}-{args.position_embedding}-"
        f"{args.opt}-lr{args.lr}-adam{args.adam_lr}-bs{args.batch_size}-"
        f"block{args.block_size}-frac{args.train_fraction:g}-dseed{args.data_seed}-"
        f"steps{args.max_steps}-warmup{args.warmup_steps}-eval{args.eval_interval}-"
        f"sched{args.lr_scheduler}-{args.tele_method}-nag{int(args.nesterov)}"
        f"{teleport_tag}-seed{args.seed}"
    )


def autocast_context(args: argparse.Namespace, device: torch.device):
    if device.type == "cuda" and args.amp_dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support bfloat16 autocast")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def sampled_logits_for_function_check(
    model: nn.Module,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    """Sample a few token-position logits without retaining the full LM logits."""
    was_training = model.training
    model.eval()
    seq_len = int(input_ids.shape[1])
    positions = sorted({0, seq_len // 2, max(0, seq_len - 1)})
    try:
        with torch.no_grad(), autocast_context(args, device):
            logits = model(input_ids=input_ids[:1]).logits
            return logits[0, positions, :].detach().float().clone()
    finally:
        model.train(was_training)


def function_preservation_metrics(before: torch.Tensor, after: torch.Tensor) -> dict[str, float]:
    difference = after.float() - before.float()
    denominator = torch.linalg.vector_norm(before.float()).clamp_min(1e-12)
    return {
        "teleport/function_relative_logit_error": float((torch.linalg.vector_norm(difference) / denominator).item()),
        "teleport/function_max_abs_logit_error": float(difference.abs().max().item()),
        "teleport/function_rms_logit_error": float(difference.square().mean().sqrt().item()),
    }


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def apply_step_learning_rates(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    step: int,
    args: argparse.Namespace,
) -> None:
    """Set warmup/cosine/step learning rates for the upcoming optimizer update."""
    warmup = int(args.warmup_steps)
    if warmup > 0 and step <= warmup:
        scale = float(step) / float(warmup)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * scale
        return

    if args.lr_scheduler == "plateau":
        # ReduceLROnPlateau owns the rates after warmup; do not overwrite them.
        return

    if args.lr_scheduler == "cosine":
        decay_steps = max(1, int(args.max_steps) - warmup)
        progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            min_lr = min(float(args.lr_min), base_lr)
            group["lr"] = min_lr + (base_lr - min_lr) * cosine
    elif args.lr_scheduler == "step":
        decay_steps = max(1, int(args.lr_step_size))
        num_decays = max(0, (step - warmup) // decay_steps)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = max(float(args.lr_min), base_lr * (float(args.lr_gamma) ** num_decays))
    else:  # none
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr


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
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.log_interval <= 0 or args.eval_interval <= 0:
        raise ValueError("log-interval and eval-interval must be positive")
    if args.max_train_batches < 0 or args.max_eval_batches < 0:
        raise ValueError("max-train-batches and max-eval-batches must be non-negative")
    if args.train_fraction <= 0.0 or args.train_fraction > 1.0:
        raise ValueError("train-fraction must be in (0, 1]")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be non-negative")
    if args.lr_min < 0.0 or args.lr_step_size <= 0 or args.lr_gamma <= 0.0:
        raise ValueError("lr-min must be non-negative; lr-step-size and lr-gamma must be positive")
    if args.tele_start_step < 1 or args.tele_limit_step < 0 or args.tele_interval < 0:
        raise ValueError("tele-start-step must be >= 1; tele-limit-step and tele-interval must be >= 0")
    diagnostic_intervals = (
        args.diagnostics_interval, args.diagnostics_spectrum_interval,
        args.diagnostics_csd_interval, args.diagnostics_tensor_interval,
    )
    if any(value < 0 for value in diagnostic_intervals):
        raise ValueError("diagnostic intervals must be non-negative")
    if args.diagnostics_snapshot_layers <= 0:
        raise ValueError("diagnostics-snapshot-layers must be positive")

    if args.tele_epoch_array is not None:
        warnings.warn(
            "--tele-epoch-array is deprecated and ignored in step-based mode; "
            "use --tele-start-step, --tele-limit-step, and --tele-interval instead.",
            stacklevel=2,
        )
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
        train_fraction=args.train_fraction,
        data_seed=args.data_seed,
    )
    data_loading_seconds = time.perf_counter() - data_started

    if len(train_loader) <= 0:
        raise RuntimeError("The training loader contains no batches")
    batches_per_epoch = len(train_loader)
    if args.max_train_batches > 0:
        batches_per_epoch = min(batches_per_epoch, args.max_train_batches)
    if args.max_steps is None:
        if args.epochs is not None:
            if args.epochs <= 0:
                raise ValueError("epochs must be positive when using the deprecated compatibility option")
            args.max_steps = int(args.epochs) * batches_per_epoch
            warnings.warn(
                f"--epochs is deprecated; converted {args.epochs} passes to {args.max_steps} optimizer steps. "
                "Use --max-steps for new experiments.",
                stacklevel=2,
            )
        else:
            args.max_steps = 1000
    elif args.epochs is not None:
        warnings.warn("Both --max-steps and --epochs were supplied; --max-steps takes precedence.", stacklevel=2)
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.warmup_steps >= args.max_steps and args.warmup_steps > 0:
        raise ValueError("warmup-steps must be smaller than max-steps")
    if args.tele_interval == 0:
        args.tele_interval = args.max_steps

    run_name = build_run_name(args)
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

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
    activation_recorder = GPT2ActivationRecorder(
        model, layer_indices=diagnostic_layer_indices(model, args.diagnostics_snapshot_layers)
    )
    optimizer = build_optimizer(model, args)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_gamma, patience=1,
            min_lr=args.lr_min,
        )
    else:
        scheduler = None
    model_setup_seconds = time.perf_counter() - model_started

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    project, entity, mode = load_wandb_settings(args)
    tokenizer.save_pretrained(save_dir)

    selected_train_tokens = int(data_sizes.get("train_selected_tokens", data_sizes.get("train_tokens", 0)))
    tokens_per_full_batch = args.batch_size * args.block_size
    print("\n========== GPT-2 WIKITEXT CONFIGURATION ==========")
    print("Device:", device)
    print("Position embedding:", args.position_embedding)
    print("Pretrained checkpoint:", args.model_name)
    print("Checkpoint load report:", load_report)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("GPT-2 hidden/layers/heads:", model.config.n_embd, model.config.n_layer, model.config.n_head)
    print("Train/validation/test sizes:", data_sizes)
    print("Effective train tokens:", selected_train_tokens)
    print("Train fraction:", args.train_fraction)
    print("Max optimizer steps:", args.max_steps)
    print("Tokens per full batch:", tokens_per_full_batch)
    print("Approx. tokens at max steps:", args.max_steps * tokens_per_full_batch)
    print("Warmup/evaluation/log intervals:", args.warmup_steps, args.eval_interval, args.log_interval)
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
                "effective_max_steps": args.max_steps,
                "effective_train_tokens": selected_train_tokens,
                "tokens_per_full_optimizer_step": tokens_per_full_batch,
                "diagnostics_enabled": bool(
                    args.diagnostics_interval or args.diagnostics_spectrum_interval
                    or args.diagnostics_csd_interval or args.diagnostics_tensor_interval
                    or args.diagnostics_artifact_on_teleport
                ),
                "diagnostics_snapshot_layers": diagnostic_layer_indices(model, args.diagnostics_snapshot_layers),
            },
            allow_val_change=True,
        )
        run.log(
            {
                "timing/data_loading_seconds": data_loading_seconds,
                "timing/model_setup_seconds": model_setup_seconds,
                "timing/initialization_seconds": time.perf_counter() - setup_started,
            },
            step=0,
        )

        attempted_teleports = 0
        accepted_teleports = 0
        global_step = 0
        total_processed_tokens = 0
        best_val_loss = float("inf")
        best_step = 0
        best_epoch = 0
        total_train_seconds = 0.0
        total_validation_seconds = 0.0

        telemetry_table = wandb.Table(
            columns=[
                "epoch", "batch", "global_step", "method", "accepted", "reason",
                "search_time_seconds", "selection_metric", "baseline_score",
                "selected_score", "score_gain", "gradient_l2_gain", "csd_l2_gain",
                "selected_candidate_index", "num_candidates_evaluated",
                "post_loss_gain", "candidate_score_post_loss_spearman", "candidate_score_margin",
                "function_relative_logit_error",
            ]
        )
        candidate_table = wandb.Table(
            columns=[
                "epoch", "batch", "global_step", "candidate_index", "is_identity",
                "selection_metric", "selection_score", "gradient_l2",
                "csd_l2_score", "csd_l2_squared_score", "csd_l2_rank",
                "scale_min", "scale_max", "transform_type", "post_step_loss",
            ]
        )

        tele_scheduler = None
        if args.tele_method in {"naive", "csd_l2_nesterov", "muon_alignment"} and args.tele_batch > 0:
            tele_scheduler = generate_tele_scheduler(
                tele_batch=args.tele_batch,
                number_of_batch=args.tele_interval,
                tele_opt=args.tele_opt,
                tele_cons=args.tele_cons,
            )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_iterator = iter(train_loader)
        epoch = 1
        batch_index_in_epoch = 0
        interval_loss_sum = 0.0
        interval_tokens = 0
        interval_steps = 0
        interval_started = time.perf_counter()

        progress = tqdm(range(1, args.max_steps + 1), desc="Training", unit="step")
        for step_number in progress:
            if batch_index_in_epoch >= batches_per_epoch:
                epoch += 1
                batch_index_in_epoch = 0
                train_iterator = iter(train_loader)
            try:
                input_ids, targets = next(train_iterator)
            except StopIteration:
                epoch += 1
                batch_index_in_epoch = 0
                train_iterator = iter(train_loader)
                try:
                    input_ids, targets = next(train_iterator)
                except StopIteration as exc:
                    raise RuntimeError("The training loader produced no batches") from exc

            batch_index = batch_index_in_epoch
            batch_index_in_epoch += 1
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            model.train()

            apply_step_learning_rates(optimizer, base_lrs, step_number, args)

            teleport_is_scheduled = (
                tele_scheduler is not None
                and step_number >= args.tele_start_step
                and (args.tele_limit_step == 0 or step_number <= args.tele_limit_step)
                and bool(tele_scheduler[(step_number - 1) % args.tele_interval])
            )
            teleport_info = None
            accepted_teleport_event = False
            pre_teleport_weights = None
            pre_teleport_sampled_logits = None
            function_check_metrics = {}
            snapshot_layers = diagnostic_layer_indices(model, args.diagnostics_snapshot_layers)
            if teleport_is_scheduled:
                # Save only representative exact matrices; full GPT-2 snapshots
                # across every layer can be very large.
                if args.diagnostics_artifact_on_teleport:
                    pre_teleport_weights = collect_weight_tensors(
                        model, layer_indices=snapshot_layers, clone=True
                    )
                if args.diagnostics_function_check:
                    pre_teleport_sampled_logits = sampled_logits_for_function_check(
                        model, input_ids, args, device
                    )
                accepted, info = try_teleportation(
                    model=model,
                    optimizer=optimizer,
                    batch=(input_ids, targets),
                    criterion=criterion,
                    args=args,
                    step_index=global_step,
                )
                teleport_info = info
                accepted_teleport_event = bool(accepted)
                if accepted and pre_teleport_sampled_logits is not None:
                    post_teleport_sampled_logits = sampled_logits_for_function_check(
                        model, input_ids, args, device
                    )
                    function_check_metrics = function_preservation_metrics(
                        pre_teleport_sampled_logits, post_teleport_sampled_logits
                    )
                    info.update(function_check_metrics)
                elif pre_teleport_sampled_logits is not None:
                    function_check_metrics = {
                        "teleport/function_relative_logit_error": 0.0,
                        "teleport/function_max_abs_logit_error": 0.0,
                        "teleport/function_rms_logit_error": 0.0,
                    }
                    info.update(function_check_metrics)

                attempted_teleports += 1
                accepted_teleports += int(accepted)
                telemetry_table.add_data(
                    epoch,
                    batch_index,
                    step_number,
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
                    info.get("post_loss_gain"),
                    info.get("candidate_score_post_loss_spearman"),
                    info.get("candidate_score_margin"),
                    info.get("teleport/function_relative_logit_error"),
                )
                for candidate in info.get("results", []):
                    candidate_table.add_data(
                        epoch,
                        batch_index,
                        step_number,
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
                        candidate.get("post_step_loss"),
                    )
                event_log = {
                    "teleport/attempted": attempted_teleports,
                    "teleport/accepted": accepted_teleports,
                    "teleport/acceptance_rate": accepted_teleports / attempted_teleports,
                    "teleport/search_time_seconds": info.get("search_time_seconds", 0.0),
                    "teleport/selection_score": info.get("selected_score", info.get("selected_csd_l2_score")),
                    "teleport/score_gain": info.get("score_gain", info.get("csd_l2_gain")),
                    "teleport/gradient_l2_gain": info.get("gradient_l2_gain"),
                    "teleport/selected_gradient_l2": info.get("selected_gradient_l2"),
                    "teleport/csd_l2_gain": info.get("csd_l2_gain"),
                    "teleport/selected_csd_l2_score": info.get("selected_csd_l2_score"),
                    "teleport/selected_candidate_index": info.get("selected_candidate_index"),
                    "teleport/accepted_event": int(accepted),
                    "teleport/post_loss_gain": info.get("post_loss_gain"),
                    "teleport/candidate_score_post_loss_spearman": info.get("candidate_score_post_loss_spearman"),
                    "teleport/candidate_score_margin": info.get("candidate_score_margin"),
                    **function_check_metrics,
                }
                run.log({key: value for key, value in event_log.items() if value is not None}, step=step_number)

            periodic_diagnostics = (
                args.diagnostics_interval > 0 and step_number % args.diagnostics_interval == 0
            )
            spectrum_diagnostics = (
                args.diagnostics_spectrum_interval > 0
                and step_number % args.diagnostics_spectrum_interval == 0
            )
            csd_diagnostics_due = (
                args.diagnostics_csd_interval > 0 and step_number % args.diagnostics_csd_interval == 0
            )
            tensor_snapshot_due = (
                args.diagnostics_tensor_interval > 0
                and step_number % args.diagnostics_tensor_interval == 0
            ) or (accepted_teleport_event and args.diagnostics_artifact_on_teleport)
            should_diagnose = (
                periodic_diagnostics or spectrum_diagnostics or csd_diagnostics_due
                or tensor_snapshot_due or teleport_is_scheduled
            )
            activation_recorder.clear()
            activation_recorder.set_enabled(should_diagnose)
            if hasattr(optimizer, "diagnostics_enabled"):
                optimizer.diagnostics_enabled = bool(should_diagnose and args.opt == "muon")

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args, device):
                logits = model(input_ids=input_ids).logits
                loss = compute_loss(logits, targets, criterion)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            diagnostic_payload = {}
            tensor_snapshot = None
            if should_diagnose:
                diagnostic_payload.update(collect_weight_geometry(
                    model,
                    include_spectrum=spectrum_diagnostics,
                    spectrum_layers=snapshot_layers,
                ))
                diagnostic_payload.update(collect_gradient_geometry(
                    model,
                    include_spectrum=spectrum_diagnostics,
                    spectrum_layers=snapshot_layers,
                ))
                diagnostic_payload.update(activation_recorder.get_metrics())
                diagnostic_payload.update(collect_global_gradient_metrics(model))
                diagnostic_payload.update({
                    "diagnostics/global_step": float(step_number),
                    "diagnostics/epoch": float(epoch),
                    "diagnostics/teleport_event": float(teleport_is_scheduled),
                    "diagnostics/accepted_teleport_event": float(accepted_teleport_event),
                })
                if csd_diagnostics_due:
                    diagnostic_payload.update(collect_csd_diagnostics(
                        model, optimizer, use_nesterov_score=bool(args.nesterov)
                    ))
                if tensor_snapshot_due:
                    extra_tensors = {}
                    if accepted_teleport_event and pre_teleport_weights is not None:
                        extra_tensors.update({
                            f"teleport/before/{name}": tensor
                            for name, tensor in pre_teleport_weights.items()
                        })
                        post_weights = collect_weight_tensors(
                            model, layer_indices=snapshot_layers, clone=True
                        )
                        extra_tensors.update({
                            f"teleport/after/{name}": tensor
                            for name, tensor in post_weights.items()
                        })
                        diagnostic_payload.update(compare_weight_snapshots(
                            pre_teleport_weights, post_weights
                        ))
                    tensor_snapshot = build_tensor_snapshot(
                        model,
                        activation_recorder,
                        snapshot_layers,
                        extra_tensors=extra_tensors,
                        include_weights=not (
                            accepted_teleport_event and pre_teleport_weights is not None
                        ),
                    )

            optimizer.step()
            if should_diagnose:
                diagnostic_payload.update(optimizer_diagnostic_metrics(optimizer))
                if diagnostic_payload:
                    run.log(diagnostic_payload, step=step_number)
                if tensor_snapshot is not None:
                    log_tensor_visuals(run, tensor_snapshot, step=step_number)
                    log_tensor_artifact(run, tensor_snapshot, save_dir, step_number, epoch)
            activation_recorder.set_enabled(False)
            if hasattr(optimizer, "diagnostics_enabled"):
                optimizer.diagnostics_enabled = False

            token_count = int(targets.numel())
            loss_value = float(loss.detach().float().item())
            interval_loss_sum += loss_value * token_count
            interval_tokens += token_count
            interval_steps += 1
            total_processed_tokens += token_count
            global_step = step_number

            progress.set_postfix(
                loss=f"{loss_value:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                epoch=epoch,
            )

            should_log = (global_step % args.log_interval == 0) or (global_step == args.max_steps)
            should_eval = (global_step % args.eval_interval == 0) or (global_step == args.max_steps)
            if should_log or should_eval:
                synchronize_device(device)
                elapsed = max(time.perf_counter() - interval_started, 1e-9)
                total_train_seconds += elapsed
                if interval_tokens > 0:
                    interval_loss = interval_loss_sum / interval_tokens
                    train_log = {
                        "train/loss": interval_loss,
                        "train/perplexity": math.exp(min(interval_loss, 20.0)),
                        "train/interval_tokens": interval_tokens,
                        "train/processed_tokens": total_processed_tokens,
                        "train/tokens_per_second": interval_tokens / elapsed,
                        "train/steps_per_second": interval_steps / elapsed,
                        "train/epoch": epoch,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    if args.opt == "muon":
                        train_log["train/muon_learning_rate"] = optimizer.param_groups[0]["lr"]
                        train_log["train/adam_learning_rate"] = optimizer.param_groups[1]["lr"]
                    else:
                        train_log["train/adamw_learning_rate"] = optimizer.param_groups[0]["lr"]
                    run.log(train_log, step=global_step)
                    interval_loss_sum = 0.0
                    interval_tokens = 0
                    interval_steps = 0
                if not should_eval:
                    interval_started = time.perf_counter()

            if should_eval:
                validation_started = time.perf_counter()
                val_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    criterion,
                    args,
                    desc=f"[Step {global_step}/{args.max_steps}] Validation",
                )
                synchronize_device(device)
                validation_seconds = time.perf_counter() - validation_started
                total_validation_seconds += validation_seconds

                if scheduler is not None and global_step >= args.warmup_steps:
                    scheduler.step(val_metrics["loss"])

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "batch_index_in_epoch": batch_index_in_epoch,
                    "global_step": global_step,
                    "processed_tokens": total_processed_tokens,
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
                    best_step = global_step
                    best_epoch = epoch
                    torch.save(checkpoint, os.path.join(save_dir, "best.pt"))

                run.log(
                    {
                        "global_step": global_step,
                        "processed_tokens": total_processed_tokens,
                        "validation/loss": val_metrics["loss"],
                        "validation/perplexity": val_metrics["perplexity"],
                        "validation/tokens": val_metrics["tokens"],
                        "validation/batches": val_metrics["batches"],
                        "best/validation_loss": best_val_loss,
                        "best/validation_perplexity": math.exp(min(best_val_loss, 20.0)),
                        "best/step": best_step,
                        "best/epoch": best_epoch,
                        "timing/validation_seconds": validation_seconds,
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
                interval_started = time.perf_counter()
                model.train()

        best_checkpoint_path = os.path.join(save_dir, "best.pt")
        if not os.path.isfile(best_checkpoint_path):
            raise RuntimeError("No best validation checkpoint was saved")
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
                "test/best_checkpoint_step": best_step,
            },
            step=global_step,
        )
        run.log({"teleport/events": telemetry_table, "teleport/candidates": candidate_table})
        run.summary["best/validation_loss"] = best_val_loss
        run.summary["best/validation_perplexity"] = math.exp(min(best_val_loss, 20.0))
        run.summary["best/step"] = best_step
        run.summary["best/epoch"] = best_epoch
        run.summary["test/perplexity"] = test_metrics["perplexity"]
        run.summary["total/optimizer_steps"] = global_step
        run.summary["total/processed_tokens"] = total_processed_tokens
        run.summary["teleport/attempted"] = attempted_teleports
        run.summary["teleport/accepted"] = accepted_teleports
        run.summary["timing/total_train_seconds"] = total_train_seconds
        run.summary["timing/total_validation_seconds"] = total_validation_seconds
        run.summary["timing/run_seconds"] = time.perf_counter() - run_started
        activation_recorder.close()


if __name__ == "__main__":
    main()
