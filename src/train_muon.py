#!/usr/bin/env python3
"""CSD-based Muon-teleportation launcher.

This launcher executes the repository's existing train.py with only its
teleport import redirected to teleport_muon.py. It does not patch the ordinary
training loop: every minibatch uses the repository's regular Muon/Adam step.
At scheduled minibatches, teleport_muon.py evaluates symmetry candidates with
a CSD predicted-decrease score and may commit one symmetry before that same
minibatch's ordinary optimizer step.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional


HERE = Path(__file__).resolve().parent
BASE_TRAIN = HERE / "train.py"
CSD_TELEPORT = HERE / "teleport_muon.py"


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


def _load_csd_teleport_module():
    if not CSD_TELEPORT.is_file():
        raise SystemExit(f"Missing CSD teleport module: {CSD_TELEPORT}")
    spec = importlib.util.spec_from_file_location("_teleport_muon", CSD_TELEPORT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load CSD teleport module: {CSD_TELEPORT}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution, as normal imports do.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_training_source(source: str) -> str:
    """Patch only the teleport import; leave all optimizer steps untouched."""
    old_import = "from teleport import generate_tele_scheduler, try_teleportation"
    new_import = "from _teleport_muon import generate_tele_scheduler, try_teleportation"
    if source.count(old_import) != 1:
        raise SystemExit(
            "train.py changed: could not find its expected teleport import exactly once. "
            "Review train-muon.py before running against a modified trainer."
        )
    return source.replace(old_import, new_import, 1)


def main() -> None:
    argv = sys.argv[1:]
    tele_method = _option_value(argv, "--tele-method")
    optimizer_name = _option_value(argv, "--opt")

    if tele_method != "muon_alignment":
        raise SystemExit(
            "CSD-Muon teleportation uses the existing --tele-method muon_alignment "
            "interface. For naive teleportation or plain training, use src/train.py."
        )
    if optimizer_name != "muon":
        raise SystemExit(
            "CSD-Muon teleportation requires --opt muon. "
            "The existing AdamW/SGD and naive paths are not handled by this trainer."
        )
    if not BASE_TRAIN.is_file():
        raise SystemExit(f"Missing base training script: {BASE_TRAIN}")

    _load_csd_teleport_module()
    original_source = BASE_TRAIN.read_text(encoding="utf-8")
    patched_source = _patch_training_source(original_source)

    namespace = {
        "__name__": "__main__",
        "__file__": str(BASE_TRAIN),
        "__package__": None,
    }
    code = compile(patched_source, str(BASE_TRAIN), "exec")
    exec(code, namespace, namespace)


if __name__ == "__main__":
    main()
