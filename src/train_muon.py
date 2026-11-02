#!/usr/bin/env python3
"""Momo-Muon-only launcher for the repository's existing ViT trainer.

Run from the repository root exactly as:
    python3 src/train-muon.py --opt muon --tele-method muon_alignment ...

To avoid changing the established train.py API or duplicating its training
implementation, this launcher executes an in-memory, narrowly patched copy of
train.py. It redirects the teleport import to teleport-muon.py and updates the
Momo scalar model after backward on every minibatch. The on-disk train.py and
teleport.py are never modified.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional


HERE = Path(__file__).resolve().parent
BASE_TRAIN = HERE / "train.py"
MOMO_TELEPORT = HERE / "teleport_muon.py"


def _option_value(argv: List[str], option: str) -> Optional[str]:
    """Return the last value supplied for an argparse-style option."""
    value = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            if index + 1 >= len(argv):
                raise SystemExit(f"{option} requires a value")
            value = argv[index + 1]
            index += 2
            continue
        prefix = option + "="
        if token.startswith(prefix):
            value = token[len(prefix):]
        index += 1
    return value


def _load_momo_teleport_module():
    if not MOMO_TELEPORT.is_file():
        raise SystemExit(f"Missing Momo teleport module: {MOMO_TELEPORT}")
    spec = importlib.util.spec_from_file_location("_teleport_muon", MOMO_TELEPORT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load Momo teleport module: {MOMO_TELEPORT}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution, as normal imports do.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_training_source(source: str) -> str:
    old_import = "from teleport import generate_tele_scheduler, try_teleportation"
    new_import = (
        "from _teleport_muon import (generate_tele_scheduler, try_teleportation, "
        "update_momo_for_step)"
    )
    if source.count(old_import) != 1:
        raise SystemExit(
            "train.py changed: could not find its expected teleport import exactly once. "
            "Review train-muon.py before running against a modified trainer."
        )
    source = source.replace(old_import, new_import, 1)

    # Expose the Momo objective lower bound as an optional CLI argument.
    candidate_arg = '    parser.add_argument("--tele-candidates", type=int, default=6)\n'
    momo_arg = (
        candidate_arg
        + '    parser.add_argument("--tele-momo-f-star", type=float, default=0.0, '
          'help="Known lower bound F_* for the Momo objective (0 for cross-entropy).")\n'
    )
    if "--tele-momo-f-star" not in source:
        if source.count(candidate_arg) != 1:
            raise SystemExit(
                "train.py changed: could not locate the --tele-candidates argument "
                "to add --tele-momo-f-star."
            )
        source = source.replace(candidate_arg, momo_arg, 1)

    # Update Momo's scalar model once per minibatch. On a teleportation batch,
    # the candidate search has already done this and update_momo_for_step's
    # last_step guard prevents a duplicate update.
    optimizer_step = "                loss.backward()\n                optimizer.step()\n"
    momo_step = (
        "                loss.backward()\n"
        '                if args.tele_method == "muon_alignment":\n'
        "                    update_momo_for_step(model, optimizer, loss, global_step, args)\n"
        "                optimizer.step()\n"
    )
    if source.count(optimizer_step) != 1:
        raise SystemExit(
            "train.py changed: could not locate the expected backward/optimizer.step block "
            "exactly once. Refusing to start without updating Momo on every minibatch."
        )
    source = source.replace(optimizer_step, momo_step, 1)
    return source


def main() -> None:
    argv = sys.argv[1:]
    tele_method = _option_value(argv, "--tele-method")
    optimizer_name = _option_value(argv, "--opt")

    if tele_method != "muon_alignment":
        raise SystemExit(
            "train-muon.py is dedicated to Momo-Muon teleportation. "
            "Pass --tele-method muon_alignment. For naive teleportation or plain "
            "training, continue using src/train.py."
        )
    if optimizer_name != "muon":
        raise SystemExit(
            "Momo-Muon teleportation requires --opt muon. "
            "The existing AdamW/SGD and naive paths are not handled by this trainer."
        )

    if not BASE_TRAIN.is_file():
        raise SystemExit(f"Missing base training script: {BASE_TRAIN}")

    _load_momo_teleport_module()
    original_source = BASE_TRAIN.read_text(encoding="utf-8")
    patched_source = _patch_training_source(original_source)

    # Keep argparse behavior and all existing training code, but give the
    # executed module the base train.py path so relative project paths remain
    # consistent with the original trainer.
    namespace = {
        "__name__": "__main__",
        "__file__": str(BASE_TRAIN),
        "__package__": None,
    }
    code = compile(patched_source, str(BASE_TRAIN), "exec")
    exec(code, namespace, namespace)


if __name__ == "__main__":
    main()
