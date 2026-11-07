"""WikiText tokenization and fixed-length next-token data loaders for GPT-2.

The default source is a local WikiText-103 directory containing train.txt,
valid.txt, and test.txt. The Kaggle .tokens filenames are also accepted.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset


_LOCAL_SPLIT_NAMES = {
    "train": ("train.txt", "wiki.train.tokens", "train"),
    "validation": ("valid.txt", "validation.txt", "wiki.valid.tokens", "valid"),
    "test": ("test.txt", "wiki.test.tokens", "test"),
}


class TokenBlockDataset(Dataset):
    """Non-overlapping input blocks with one-token-ahead targets."""

    def __init__(self, token_ids: torch.Tensor, block_size: int):
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional tensor")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.token_ids = token_ids.to(device="cpu", dtype=torch.int32)
        self.block_size = int(block_size)
        self.num_blocks = max(0, (self.token_ids.numel() - 1) // self.block_size)
        if self.num_blocks == 0:
            raise ValueError(
                f"Not enough tokens ({self.token_ids.numel()}) for block_size={block_size}"
            )

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(index)
        start = index * self.block_size
        inputs = self.token_ids[start : start + self.block_size].to(torch.long)
        targets = self.token_ids[start + 1 : start + self.block_size + 1].to(torch.long)
        return inputs, targets


def load_gpt2_tokenizer(model_name: str = "gpt2", cache_dir: Optional[str] = None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("Install transformers to load the GPT-2 tokenizer") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("The selected tokenizer has no EOS token")
    return tokenizer


def _find_local_split_files(data_dir: str) -> dict[str, str]:
    """Find the three raw text splits, allowing common Kaggle layouts."""
    root = Path(data_dir).expanduser()
    search_roots = (
        root,
        root / "wikitext-103",
        root / "wikitext-103" / "wikitext-103",
    )
    resolved: dict[str, str] = {}
    for split, candidates in _LOCAL_SPLIT_NAMES.items():
        found = None
        for candidate_root in search_roots:
            for filename in candidates:
                candidate = candidate_root / filename
                if candidate.is_file():
                    found = str(candidate.resolve())
                    break
            if found:
                break
        if not found:
            expected = ", ".join(_LOCAL_SPLIT_NAMES[split])
            raise FileNotFoundError(
                f"Cannot find the WikiText {split} split under '{root}'. "
                f"Expected one of: {expected}. The canonical files are "
                "train.txt, valid.txt, and test.txt."
            )
        resolved[split] = found
    return resolved


def _cache_path(
    data_dir: str,
    dataset_config: str,
    model_name: str,
    source: str,
    source_files: Optional[dict[str, str]] = None,
) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_config)
    signature = [source, dataset_config, model_name]
    if source_files:
        for split in ("train", "validation", "test"):
            path = source_files[split]
            stat = os.stat(path)
            signature.append(
                f"{split}:{os.path.basename(path)}:{stat.st_size}:{stat.st_mtime_ns}"
            )
    digest = hashlib.sha256("|".join(signature).encode("utf-8")).hexdigest()[:16]
    return os.path.join(
        data_dir, "token_cache", f"{safe_dataset}_{safe_model}_{source}_{digest}.pt"
    )


def _encode_text_file(
    path: str,
    tokenizer: Any,
    *,
    lines_per_chunk: int = 1024,
) -> torch.Tensor:
    """Encode a large raw text split without reading it all as one string."""
    pieces: list[torch.Tensor] = []
    line_buffer: list[str] = []

    def flush_buffer() -> None:
        if not line_buffer:
            return
        text = "".join(line_buffer)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            pieces.append(torch.tensor(ids, dtype=torch.int32))
        line_buffer.clear()

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as file:
        for line in file:
            line_buffer.append(line)
            if len(line_buffer) >= lines_per_chunk:
                flush_buffer()
        flush_buffer()

    if not pieces:
        raise ValueError(f"WikiText split '{path}' produced no tokens")
    token_ids = torch.cat(pieces, dim=0)
    eos_id = int(tokenizer.eos_token_id)
    if int(token_ids[-1]) != eos_id:
        token_ids = torch.cat((token_ids, torch.tensor([eos_id], dtype=torch.int32)))
    return token_ids


def _encode_hf_split(split: Any, tokenizer: Any) -> torch.Tensor:
    rows = split["text"]
    text = "\n".join(str(row) for row in rows)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("A WikiText split produced no tokens")
    if int(ids[-1]) != int(tokenizer.eos_token_id):
        ids.append(int(tokenizer.eos_token_id))
    return torch.tensor(ids, dtype=torch.int32)


def load_wikitext_token_ids(
    dataset_config: str,
    tokenizer: Any,
    data_dir: str,
    cache_dir: Optional[str] = None,
    force_retokenize: bool = False,
    dataset_source: str = "local",
) -> dict[str, torch.Tensor]:
    """Load local WikiText-103 text files, or optionally a HF WikiText split."""
    if dataset_source not in {"local", "huggingface"}:
        raise ValueError("dataset_source must be 'local' or 'huggingface'")

    source_files = _find_local_split_files(data_dir) if dataset_source == "local" else None
    cache_path = _cache_path(
        data_dir,
        dataset_config,
        tokenizer.name_or_path,
        dataset_source,
        source_files,
    )
    if os.path.isfile(cache_path) and not force_retokenize:
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            expected = {"train", "validation", "test"}
            if (
                isinstance(cached, dict)
                and expected.issubset(cached)
                and all(torch.is_tensor(cached[key]) for key in expected)
            ):
                return {key: cached[key].to(dtype=torch.int32) for key in expected}
        except (OSError, RuntimeError, ValueError, EOFError):
            pass

    if dataset_source == "local":
        assert source_files is not None
        tokenized = {
            split: _encode_text_file(path, tokenizer)
            for split, path in source_files.items()
        }
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError(
                "Hugging Face dataset loading requires the 'datasets' package. "
                "Install requirements.txt or use --dataset-source local."
            ) from exc
        raw = load_dataset("wikitext", dataset_config, cache_dir=cache_dir)
        tokenized = {
            "train": _encode_hf_split(raw["train"], tokenizer),
            "validation": _encode_hf_split(raw["validation"], tokenizer),
            "test": _encode_hf_split(raw["test"], tokenizer),
        }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(tokenized, cache_path)
    return tokenized


def get_wikitext_dataloaders(
    *,
    dataset_config: str,
    model_name: str,
    block_size: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    data_dir: str,
    cache_dir: Optional[str] = None,
    force_retokenize: bool = False,
    pin_memory: bool = True,
    dataset_source: str = "local",
    train_fraction: float = 1.0,
    data_seed: int = 42,
):
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")

    tokenizer = load_gpt2_tokenizer(model_name, cache_dir)
    token_splits = load_wikitext_token_ids(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        data_dir=data_dir,
        cache_dir=cache_dir,
        force_retokenize=force_retokenize,
        dataset_source=dataset_source,
    )
    datasets = {
        split: TokenBlockDataset(tokens, block_size)
        for split, tokens in token_splits.items()
    }

    full_train_dataset = datasets["train"]
    if train_fraction < 1.0:
        n_train_blocks = max(1, int(len(full_train_dataset) * train_fraction))
        generator = torch.Generator()
        generator.manual_seed(int(data_seed))
        indices = torch.randperm(
            len(full_train_dataset), generator=generator
        )[:n_train_blocks].tolist()
        datasets["train"] = Subset(full_train_dataset, indices)

    common = {
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory and torch.cuda.is_available()),
        "persistent_workers": bool(num_workers > 0),
    }
    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=True,
        drop_last=False, **common
    )
    validation_loader = DataLoader(
        datasets["validation"], batch_size=eval_batch_size, shuffle=False,
        drop_last=False, **common
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=eval_batch_size, shuffle=False,
        drop_last=False, **common
    )

    sizes = {f"{split}_tokens": int(tokens.numel()) for split, tokens in token_splits.items()}
    sizes.update({f"{split}_blocks": len(dataset) for split, dataset in datasets.items()})
    sizes["train_selected_tokens"] = int(len(datasets["train"]) * block_size)
    sizes["train_selected_blocks"] = int(len(datasets["train"]))
    sizes["train_fraction"] = float(train_fraction)
    sizes["train_data_seed"] = int(data_seed)
    return train_loader, validation_loader, test_loader, tokenizer, sizes
