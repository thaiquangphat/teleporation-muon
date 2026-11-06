#!/usr/bin/env python3
"""Patch src/lm/train.py for the split LM package and Nesterov CSD-L2.

Run once from the repository root after extracting the companion replacement
files. The script intentionally checks every old code block before changing it.
"""

from __future__ import annotations

from pathlib import Path


TRAIN_PATH = Path("src/lm/train.py")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count == 0:
        if new in source:
            return source  # already patched; make this script idempotent
        raise RuntimeError(f"Could not find expected block: {label}")
    if count != 1:
        raise RuntimeError(f"Expected exactly one match for {label}, found {count}")
    return source.replace(old, new, 1)


def main() -> None:
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(
            f"Run this script from the repository root; missing {TRAIN_PATH}"
        )

    source = TRAIN_PATH.read_text(encoding="utf-8")

    source = replace_once(
        source,
        '''try:
    from .lm_datamodule import get_wikitext_dataloaders
    from .lm_model import GPT2LMForCausalLM
    from .lm_teleport_muon import (
        create_lm_muon_optimizer,
        generate_tele_scheduler,
        try_teleportation,
    )
except ImportError:
    from lm_datamodule import get_wikitext_dataloaders
    from lm_model import GPT2LMForCausalLM
    from lm_teleport_muon import (
        create_lm_muon_optimizer,
        generate_tele_scheduler,
        try_teleportation,
    )''',
        '''try:
    from .datamodule import get_wikitext_dataloaders
    from .model import GPT2LMForCausalLM
    from .teleport import create_lm_muon_optimizer, generate_tele_scheduler
    from .csd_nesterov import try_teleportation
except ImportError:  # python src/lm/train.py
    from datamodule import get_wikitext_dataloaders
    from model import GPT2LMForCausalLM
    from teleport import create_lm_muon_optimizer, generate_tele_scheduler
    from csd_nesterov import try_teleportation''',
        "LM module imports",
    )

    source = replace_once(
        source,
        '    parser.add_argument("--data-dir", type=str, default="./data")',
        '    parser.add_argument("--data-dir", type=str, default="./data/wt103")\n'
        '    parser.add_argument("--dataset-source", type=str, default="local", choices=["local", "huggingface"])',
        "local WikiText data-dir and source arguments",
    )
    source = replace_once(
        source,
        '    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")',
        '    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")',
        "WikiText-103 config default",
    )
    source = replace_once(
        source,
        '    parser.add_argument("--tele-method", type=str, default="none", choices=["none", "muon_alignment"])',
        '    parser.add_argument("--tele-method", type=str, default="none", choices=["none", "csd_l2_nesterov", "muon_alignment"])',
        "Nesterov CSD method CLI choice",
    )
    source = replace_once(
        source,
        '    parser.add_argument("--tele-accept-tol", type=float, default=1e-6)\n',
        "",
        "remove obsolete CSD acceptance tolerance",
    )

    source = replace_once(
        source,
        '''    if args.tele_method == "muon_alignment" and args.opt != "muon":
        raise ValueError("CSD-Muon teleportation requires --opt muon")
    if args.tele_method == "muon_alignment" and not (args.tele_att or args.tele_mlp):
        raise ValueError("Enable --tele-att 1 and/or --tele-mlp 1 for CSD teleportation")''',
        '''    if args.tele_method in {"csd_l2_nesterov", "muon_alignment"} and args.opt != "muon":
        raise ValueError("Nesterov CSD-L2 teleportation requires --opt muon")
    if args.tele_method in {"csd_l2_nesterov", "muon_alignment"} and not (args.tele_att or args.tele_mlp):
        raise ValueError("Enable --tele-att 1 and/or --tele-mlp 1 for CSD teleportation")''',
        "CSD argument validation",
    )

    source = replace_once(
        source,
        '''        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        force_retokenize=args.force_retokenize,''',
        '''        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        force_retokenize=args.force_retokenize,
        dataset_source=args.dataset_source,''',
        "dataset source passed to LM datamodule",
    )

    source = replace_once(
        source,
        '            if args.tele_method == "muon_alignment" and epoch in args.tele_epoch_array:',
        '            if args.tele_method in {"csd_l2_nesterov", "muon_alignment"} and epoch in args.tele_epoch_array:',
        "Nesterov CSD epoch scheduler condition",
    )

    source = replace_once(
        source,
        '''                "search_time_seconds", "alignment_gain", "selected_alignment",
                "selected_candidate_index", "num_candidates_evaluated",''',
        '''                "search_time_seconds", "csd_l2_gain", "selected_csd_l2_score",
                "selected_candidate_index", "num_candidates_evaluated",''',
        "telemetry event table CSD columns",
    )
    source = replace_once(
        source,
        '''                "alignment", "alignment_rank", "scale_min", "scale_max", "transform_type",''',
        '''                "csd_l2_score", "csd_l2_squared_score", "csd_l2_rank",
                "scale_min", "scale_max", "transform_type",''',
        "telemetry candidate table CSD columns",
    )
    source = replace_once(
        source,
        '                        info.get("method", "csd_alignment"),',
        '                        info.get("method", "csd_l2_nesterov"),',
        "telemetry method fallback",
    )
    source = replace_once(
        source,
        '''                        info.get("alignment_gain"),
                        info.get("selected_alignment"),''',
        '''                        info.get("csd_l2_gain"),
                        info.get("selected_csd_l2_score"),''',
        "telemetry event CSD score values",
    )
    source = replace_once(
        source,
        '''                            candidate.get("alignment"),
                            candidate.get("alignment_rank"),''',
        '''                            candidate.get("csd_l2_score"),
                            candidate.get("csd_l2_squared_score"),
                            candidate.get("csd_l2_rank"),''',
        "telemetry candidate CSD score values",
    )
    source = replace_once(
        source,
        '''                            "teleport/alignment_gain": info.get("alignment_gain"),
                            "teleport/selected_alignment": info.get("selected_alignment"),''',
        '''                            "teleport/csd_l2_gain": info.get("csd_l2_gain"),
                            "teleport/selected_csd_l2_score": info.get("selected_csd_l2_score"),''',
        "W&B scalar CSD score keys",
    )

    # Keep the original teleport.py public entry points consistent too. The
    # imports are intentionally lazy: csd_nesterov.py reuses LM symmetry and
    # optimizer-state helpers from teleport.py.
    teleport_path = Path("src/lm/teleport.py")
    if not teleport_path.is_file():
        raise FileNotFoundError(f"Missing {teleport_path}")
    teleport_source = teleport_path.read_text(encoding="utf-8")

    old_start = teleport_source.find("def try_muon_alignment_teleportation(")
    old_middle = teleport_source.find("def try_teleportation(", old_start)
    old_end = teleport_source.find("__all__", old_middle)
    if old_start < 0 or old_middle < 0 or old_end < 0:
        if 'def try_teleportation(' not in teleport_source:
            raise RuntimeError("Could not locate public LM teleportation entry points")
        if 'return try_csd_l2_nesterov_teleportation(' not in teleport_source:
            raise RuntimeError("teleport.py entry points do not match expected source")
    else:
        new_public_api = '''def try_muon_alignment_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Backward-compatible alias for Nesterov CSD-L2 teleportation."""
    try:
        from .csd_nesterov import try_csd_l2_nesterov_teleportation
    except ImportError:  # python src/lm/train.py
        from csd_nesterov import try_csd_l2_nesterov_teleportation
    return try_csd_l2_nesterov_teleportation(
        model, optimizer, batch, criterion, args, step_index
    )


def try_teleportation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion,
    args: Any,
    step_index: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Public dispatcher; CSD aliases always use the Nesterov CSD-L2 score."""
    method = getattr(args, "tele_method", "none")
    if method in {"none", None}:
        return False, {"method": "none", "accepted": False, "step": int(step_index)}
    if method in {"csd_l2_nesterov", "muon_alignment"}:
        return try_muon_alignment_teleportation(
            model, optimizer, batch, criterion, args, step_index
        )
    raise ValueError(
        f"Unsupported LM teleportation method {method!r}; "
        "use 'none' or 'csd_l2_nesterov'."
    )
'''
        teleport_source = (
            teleport_source[:old_start]
            + new_public_api
            + "\n\n"
            + teleport_source[old_end:]
        )
        teleport_source_to_write = teleport_source
    if old_start >= 0 and old_middle >= 0 and old_end >= 0:
        teleport_path.write_text(teleport_source_to_write, encoding="utf-8")
        print(f"Patched {teleport_path}")

    TRAIN_PATH.write_text(source, encoding="utf-8")
    print(f"Patched {TRAIN_PATH}")
    print("Use --tele-method csd_l2_nesterov for the canonical LM method.")


if __name__ == "__main__":
    main()
