"""WikiText tokenization and fixed-length next-token data loaders for GPT-2."""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset


class TokenBlockDataset(Dataset):
    """Non-overlapping input blocks with one-token-ahead targets."""

    def __init__(self, token_ids: torch.Tensor, block_size: int):
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional tensor")
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.token_ids = token_ids.to(dtype=torch.long, device="cpu")
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
        inputs = self.token_ids[start : start + self.block_size]
        targets = self.token_ids[start + 1 : start + self.block_size + 1]
        return inputs, targets


def load_gpt2_tokenizer(model_name: str = "gpt2", cache_dir: Optional[str] = None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("Install transformers to load the GPT-2 tokenizer") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    # Batches are fixed-length and do not need padding, but setting this makes
    # the tokenizer safe to reuse for generation or padded inference later.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("The selected GPT-2 tokenizer has no EOS token")
    return tokenizer


def _token_cache_path(data_dir: str, dataset_config: str, model_name: str) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_config)
    return os.path.join(data_dir, "token_cache", f"{safe_dataset}_{safe_model}.pt")


def _encode_split(split: Any, tokenizer: Any) -> torch.Tensor:
    # Keep paragraph boundaries and headings; WikiText's raw split is a list
    # of text rows, including blank rows between document sections.
    rows = split["text"]
    text = "\n\n".join(str(row).strip("\n") for row in rows)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("A WikiText split produced no tokens")
    if token_ids[-1] != tokenizer.eos_token_id:
        token_ids.append(int(tokenizer.eos_token_id))
    return torch.tensor(token_ids, dtype=torch.long)


def load_wikitext_token_ids(
    dataset_config: str,
    tokenizer: Any,
    data_dir: str,
    cache_dir: Optional[str] = None,
    force_retokenize: bool = False,
) -> dict[str, torch.Tensor]:
    """Load WikiText-2 raw splits and cache GPT-2 BPE token IDs locally."""
    cache_path = _token_cache_path(data_dir, dataset_config, tokenizer.name_or_path)
    if os.path.isfile(cache_path) and not force_retokenize:
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        expected = {"train", "validation", "test"}
        if isinstance(cached, dict) and expected.issubset(cached):
            return {key: cached[key].long() for key in expected}

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "WikiText loading requires the 'datasets' package. Install "
            "requirements-lm.txt."
        ) from exc

    raw = load_dataset("wikitext", dataset_config, cache_dir=cache_dir)
    tokenized = {
        "train": _encode_split(raw["train"], tokenizer),
        "validation": _encode_split(raw["validation"], tokenizer),
        "test": _encode_split(raw["test"], tokenizer),
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
):
    tokenizer = load_gpt2_tokenizer(model_name, cache_dir)
    token_splits = load_wikitext_token_ids(
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        data_dir=data_dir,
        cache_dir=cache_dir,
        force_retokenize=force_retokenize,
    )
    datasets = {
        split: TokenBlockDataset(tokens, block_size)
        for split, tokens in token_splits.items()
    }
    common = {
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory and torch.cuda.is_available()),
        "persistent_workers": bool(num_workers > 0),
    }
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    sizes = {
        f"{split}_tokens": int(tokens.numel()) for split, tokens in token_splits.items()
    }
    sizes.update({f"{split}_blocks": len(dataset) for split, dataset in datasets.items()})
    return train_loader, validation_loader, test_loader, tokenizer, sizes
