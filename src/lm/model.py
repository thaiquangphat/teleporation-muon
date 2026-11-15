"""Configurable GPT-2-style causal language model for WikiText experiments.

The implementation intentionally mirrors ``src/vit/model.py`` at the block
level: attention has separate Q/K/V projections and an output projection,
followed by a two-layer MLP. Unlike ViT, attention remains causal.

Supported positional encodings:
- ``learned_ape`` (``ape`` is retained as an alias)
- ``sinusoidal_ape``
- ``rope``

The default configuration is the compact WikiText architecture used by the
reference experiment: 16 layers, width 128, 8 heads, and FFN width 2048.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn


_POSITION_ALIASES = {
    "ape": "learned_ape",
    "learned": "learned_ape",
    "learnable": "learned_ape",
    "learned_ape": "learned_ape",
    "sinusoidal": "sinusoidal_ape",
    "sinusoidal_ape": "sinusoidal_ape",
    "rope": "rope",
}


def normalize_position_embedding(value: str) -> str:
    try:
        return _POSITION_ALIASES[str(value).lower()]
    except KeyError as exc:
        raise ValueError(
            "position_embedding must be one of: learned_ape, sinusoidal_ape, rope"
        ) from exc


def create_sinusoidal_positions(n_positions: int, dim: int) -> torch.Tensor:
    """Build a fixed, interleaved sin/cos absolute position table."""
    if n_positions <= 0 or dim <= 0:
        raise ValueError("n_positions and dim must be positive")
    positions = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / float(dim))
    )
    angles = positions * frequencies.unsqueeze(0)
    table = torch.zeros(n_positions, dim, dtype=torch.float32)
    table[:, 0::2] = torch.sin(angles)
    if dim > 1:
        table[:, 1::2] = torch.cos(angles[:, : table[:, 1::2].shape[1]])
    return table


@dataclass
class GPT2LMConfig:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 128
    n_layer: int = 16
    n_head: int = 8
    n_inner: Optional[int] = 2048
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    activation_function: str = "relu"
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False

    @classmethod
    def from_hf_config(cls, config: Any) -> "GPT2LMConfig":
        return cls(
            vocab_size=int(config.vocab_size),
            n_positions=int(getattr(config, "n_positions", 1024)),
            n_embd=int(config.n_embd),
            n_layer=int(config.n_layer),
            n_head=int(config.n_head),
            n_inner=(int(config.n_inner) if getattr(config, "n_inner", None) else None),
            embd_pdrop=float(getattr(config, "embd_pdrop", 0.1)),
            attn_pdrop=float(getattr(config, "attn_pdrop", 0.0)),
            resid_pdrop=float(getattr(config, "resid_pdrop", 0.1)),
            layer_norm_epsilon=float(getattr(config, "layer_norm_epsilon", 1e-5)),
            initializer_range=float(getattr(config, "initializer_range", 0.02)),
            activation_function=str(getattr(config, "activation_function", "gelu_new")),
            scale_attn_weights=bool(getattr(config, "scale_attn_weights", True)),
            scale_attn_by_inverse_layer_idx=bool(
                getattr(config, "scale_attn_by_inverse_layer_idx", False)
            ),
        )


class CausalLMOutput:
    """Small Hugging Face-like output object."""

    def __init__(self, logits: torch.Tensor, loss: Optional[torch.Tensor] = None):
        self.logits = logits
        self.loss = loss


class GPT2SelfAttention(nn.Module):
    """Multi-head causal self-attention with separately addressable projections."""

    def __init__(self, config: GPT2LMConfig, layer_idx: int, position_embedding: str):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = int(config.n_head)
        self.head_dim = int(config.n_embd // config.n_head)
        if position_embedding == "rope" and self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")

        self.position_embedding = position_embedding
        self.layer_idx = int(layer_idx)
        self.scale_attn_weights = bool(config.scale_attn_weights)
        self.scale_attn_by_inverse_layer_idx = bool(config.scale_attn_by_inverse_layer_idx)

        self.query = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.diagnostics_enabled = False
        self.last_diagnostics: Optional[dict[str, torch.Tensor]] = None

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = value.shape
        return value.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    def _apply_rope(
        self, value: torch.Tensor, position_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch_size, _, seq_len, _ = value.shape
        if position_ids is None:
            positions = torch.arange(seq_len, device=value.device, dtype=torch.float32)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        else:
            positions = position_ids.to(device=value.device, dtype=torch.float32)
            if positions.ndim == 1:
                positions = positions.unsqueeze(0).expand(batch_size, -1)

        pair_index = torch.arange(0, self.head_dim, 2, device=value.device, dtype=torch.float32)
        inv_freq = torch.pow(10000.0, -pair_index / self.head_dim)
        angles = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        cos = torch.repeat_interleave(angles.cos().to(value.dtype), 2, dim=-1).unsqueeze(1)
        sin = torch.repeat_interleave(angles.sin().to(value.dtype), 2, dim=-1).unsqueeze(1)
        rotated = torch.stack((-value[..., 1::2], value[..., 0::2]), dim=-1).flatten(-2)
        return value * cos + rotated * sin

    def _attention_scale(self) -> float:
        scale = self.head_dim ** -0.5 if self.scale_attn_weights else 1.0
        if self.scale_attn_by_inverse_layer_idx:
            scale /= float(self.layer_idx + 1)
        return scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self._split_heads(self.query(hidden_states))
        key = self._split_heads(self.key(hidden_states))
        value = self._split_heads(self.value(hidden_states))
        query_pre_pe, key_pre_pe = query, key

        if self.position_embedding == "rope":
            query = self._apply_rope(query, position_ids)
            key = self._apply_rope(key, position_ids)

        desired_scale = self._attention_scale()
        # SDPA applies head_dim**(-1/2) internally. Rescale Q so that its
        # effective scale is exactly ``desired_scale`` on supported PyTorch
        # versions without relying on the optional SDPA ``scale`` argument.
        query_for_attention = query * (desired_scale * math.sqrt(self.head_dim))
        dropout_p = self.attn_dropout.p if self.training else 0.0

        if attention_mask is None:
            context = F.scaled_dot_product_attention(
                query_for_attention,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            # True denotes an allowed key position for PyTorch SDPA masks.
            causal = torch.ones(
                seq_len, seq_len, device=hidden_states.device, dtype=torch.bool
            ).tril().view(1, 1, seq_len, seq_len)
            allowed = causal & attention_mask.to(torch.bool).view(batch_size, 1, 1, seq_len)
            context = F.scaled_dot_product_attention(
                query_for_attention,
                key,
                value,
                attn_mask=allowed,
                dropout_p=dropout_p,
                is_causal=False,
            )

        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_head * self.head_dim
        )
        output = self.out_proj(context)

        if self.diagnostics_enabled:
            qk_logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * desired_scale
            causal = torch.ones(
                seq_len, seq_len, device=hidden_states.device, dtype=torch.bool
            ).tril().view(1, 1, seq_len, seq_len)
            if attention_mask is not None:
                causal = causal & attention_mask.to(torch.bool).view(batch_size, 1, 1, seq_len)
            qk_logits = qk_logits.masked_fill(~causal, torch.finfo(qk_logits.dtype).min)
            probs = torch.softmax(qk_logits, dim=-1)
            self.last_diagnostics = {
                "Q_PRE_PE": query_pre_pe.detach(),
                "K_PRE_PE": key_pre_pe.detach(),
                "Q": query.detach(),
                "K": key.detach(),
                "V": value.detach(),
                "QK_LOGITS": qk_logits.detach(),
                "ATTENTION_PROBS": probs.detach(),
                "O_INPUT": context.detach(),
                "O_OUTPUT": output.detach(),
            }
        else:
            self.last_diagnostics = None

        return self.resid_dropout(output)


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2LMConfig):
        super().__init__()
        inner_dim = int(config.n_inner or (4 * config.n_embd))
        self.c_fc = nn.Linear(config.n_embd, inner_dim, bias=True)
        self.c_proj = nn.Linear(inner_dim, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.activation_function = str(config.activation_function).lower()
        self.diagnostics_enabled = False
        self.last_diagnostics = None

    def _activation(self, value: torch.Tensor) -> torch.Tensor:
        name = self.activation_function
        if name == "relu":
            return F.relu(value)
        if name == "gelu":
            return F.gelu(value)
        if name == "gelu_new":
            return 0.5 * value * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (value + 0.044715 * torch.pow(value, 3))
                )
            )
        if name == "silu":
            return F.silu(value)
        raise ValueError(f"Unsupported activation_function: {name}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pre_activation = self.c_fc(hidden_states)
        activation = self._activation(pre_activation)
        projected = self.c_proj(activation)
        if self.diagnostics_enabled:
            self.last_diagnostics = {
                "INPUT": hidden_states.detach(),
                "FC_PRE_ACTIVATION": pre_activation.detach(),
                "ACTIVATION": activation.detach(),
                "OUTPUT_PROJECTION": projected.detach(),
            }
        else:
            self.last_diagnostics = None
        return self.dropout(projected)


class GPT2Block(nn.Module):
    """Pre-LN decoder block, retaining GPT-2's causal residual structure."""

    def __init__(self, config: GPT2LMConfig, layer_idx: int, position_embedding: str):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2SelfAttention(config, layer_idx, position_embedding)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class GPT2Transformer(nn.Module):
    def __init__(self, config: GPT2LMConfig, position_embedding: str):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = (
            nn.Embedding(config.n_positions, config.n_embd)
            if position_embedding == "learned_ape"
            else None
        )
        if position_embedding == "sinusoidal_ape":
            self.register_buffer(
                "sinusoidal_positions",
                create_sinusoidal_positions(config.n_positions, config.n_embd),
                persistent=False,
            )
        else:
            self.register_buffer("sinusoidal_positions", None, persistent=False)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPT2Block(config, i, position_embedding) for i in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def _position_values(self, position_ids: torch.Tensor) -> torch.Tensor:
        if self.wpe is not None:
            return self.wpe(position_ids)
        if self.sinusoidal_positions is None:
            return torch.zeros(
                *position_ids.shape, self.wte.embedding_dim,
                device=position_ids.device, dtype=self.wte.weight.dtype,
            )
        table = self.sinusoidal_positions
        if torch.any(position_ids < 0) or torch.any(position_ids >= table.shape[0]):
            raise ValueError(
                f"Position id outside configured range [0, {table.shape[0] - 1}]"
            )
        return table[position_ids]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden_states = self.wte(input_ids)
        if self.wpe is not None or self.sinusoidal_positions is not None:
            hidden_states = hidden_states + self._position_values(position_ids)
        hidden_states = self.drop(hidden_states)
        for block in self.h:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return self.ln_f(hidden_states)


class GPT2LMForCausalLM(nn.Module):
    """GPT-2-style causal LM with configurable dimensions and positional encoding."""

    def __init__(self, config: GPT2LMConfig, position_embedding: str = "sinusoidal_ape"):
        super().__init__()
        position_embedding = normalize_position_embedding(position_embedding)
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if config.n_positions <= 0:
            raise ValueError("n_positions must be positive")
        if position_embedding == "rope" and (config.n_embd // config.n_head) % 2:
            raise ValueError("RoPE requires an even attention head dimension")

        self.config = config
        self.position_embedding = position_embedding
        self.transformer = GPT2Transformer(config, position_embedding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tie_weights()
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        hidden_states = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "gpt2",
        position_embedding: str = "learned_ape",
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> tuple["GPT2LMForCausalLM", dict[str, Any]]:
        """Load a compatible Hugging Face GPT-2 checkpoint.

        The fused Hugging Face ``c_attn`` projection is split into the explicit
        Q/K/V projections used by this model. This path is optional; the
        paper-style WikiText configuration is trained from scratch.
        """
        try:
            from transformers import GPT2LMHeadModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Pretrained initialization requires transformers. Install requirements.txt."
            ) from exc

        load_kwargs: dict[str, Any] = {"cache_dir": cache_dir}
        if revision is not None:
            load_kwargs["revision"] = revision
        reference = GPT2LMHeadModel.from_pretrained(model_name, **load_kwargs)
        config = GPT2LMConfig.from_hf_config(reference.config)
        model = cls(config, position_embedding=position_embedding)
        source_state = reference.state_dict()
        target_state = model.state_dict()
        converted: dict[str, torch.Tensor] = {}
        missing: list[str] = []

        qkv_pattern = re.compile(
            r"^transformer\.h\.(\d+)\.attn\.(query|key|value)\.(weight|bias)$"
        )
        for target_name, target_tensor in target_state.items():
            match = qkv_pattern.match(target_name)
            if match:
                layer_idx, projection_name, suffix = match.groups()
                source_name = f"transformer.h.{layer_idx}.attn.c_attn.{suffix}"
                source_tensor = source_state.get(source_name)
                if source_tensor is None:
                    missing.append(target_name)
                    continue
                if suffix == "weight":
                    source_tensor = source_tensor.transpose(0, 1).contiguous()
                chunks = source_tensor.chunk(3, dim=0)
                projection_index = {"query": 0, "key": 1, "value": 2}[projection_name]
                source_tensor = chunks[projection_index].contiguous()
            elif ".attn.out_proj." in target_name:
                source_name = target_name.replace(".attn.out_proj.", ".attn.c_proj.")
                source_tensor = source_state.get(source_name)
                if source_tensor is not None and target_name.endswith(".weight"):
                    source_tensor = source_tensor.transpose(0, 1).contiguous()
            else:
                source_tensor = source_state.get(target_name)
                if source_tensor is not None and target_name.endswith((".mlp.c_fc.weight", ".mlp.c_proj.weight")):
                    source_tensor = source_tensor.transpose(0, 1).contiguous()
                if source_tensor is None and target_name == "lm_head.weight":
                    source_tensor = source_state.get("transformer.wte.weight")

            # Fixed sinusoidal APE and RoPE deliberately do not load GPT-2's
            # learned position table.
            if source_tensor is None:
                missing.append(target_name)
                continue
            if source_tensor.shape != target_tensor.shape:
                raise RuntimeError(
                    f"Checkpoint shape mismatch for {target_name}: source "
                    f"{tuple(source_tensor.shape)}, target {tuple(target_tensor.shape)}"
                )
            converted[target_name] = source_tensor

        if missing:
            raise RuntimeError(
                "The GPT-2 checkpoint is missing required tensors: "
                + ", ".join(missing[:12])
            )
        model.load_state_dict(converted, strict=True)
        model.tie_weights()
        del reference
        report = {
            "checkpoint": model_name,
            "position_embedding": model.position_embedding,
            "loaded_tensor_count": len(converted),
            "dropped_learned_position_embedding": model.position_embedding != "learned_ape",
            "pretrained_weights_loaded": True,
        }
        return model, report

    def config_dict(self) -> dict[str, Any]:
        result = asdict(self.config)
        result["position_embedding"] = self.position_embedding
        return result
