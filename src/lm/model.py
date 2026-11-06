"""GPT-2 causal language model with APE or RoPE positional encoding.

The module uses ordinary ``nn.Linear`` layers (weight layout [out, in]) so the
repository's Muon optimizer sees the same matrix orientation as it does for
ViT. Hugging Face GPT-2 Conv1D projection weights are transposed on import.

The APE variant is functionally the standard GPT-2 decoder. The RoPE variant
reuses the pretrained GPT-2 token/transformer weights but omits GPT-2's learned
absolute position embedding and applies rotary embeddings to Q/K instead.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPT2LMConfig:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: Optional[int] = None
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    activation_function: str = "gelu_new"
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
            attn_pdrop=float(getattr(config, "attn_pdrop", 0.1)),
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
    """Small HF-like output object; exposes ``loss`` and ``logits`` attributes."""

    def __init__(self, logits: torch.Tensor, loss: Optional[torch.Tensor] = None):
        self.logits = logits
        self.loss = loss


class GPT2SelfAttention(nn.Module):
    def __init__(self, config: GPT2LMConfig, layer_idx: int, position_embedding: str):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        if position_embedding == "rope" and self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.position_embedding = position_embedding
        self.layer_idx = layer_idx
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = value.shape
        return value.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    def _apply_rope(self, value: torch.Tensor, position_ids: Optional[torch.Tensor]) -> torch.Tensor:
        # Adjacent coordinate pairs (x_0,x_1), (x_2,x_3), ... are rotated.
        batch_size, _, seq_len, _ = value.shape
        if position_ids is None:
            positions = torch.arange(seq_len, device=value.device, dtype=torch.float32)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        else:
            positions = position_ids.to(device=value.device, dtype=torch.float32)
            if positions.ndim == 1:
                positions = positions.unsqueeze(0).expand(batch_size, -1)

        pair_index = torch.arange(
            0, self.head_dim, 2, device=value.device, dtype=torch.float32
        )
        inv_freq = torch.pow(10000.0, -pair_index / self.head_dim)
        angles = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        cos = angles.cos().to(dtype=value.dtype).unsqueeze(1)
        sin = angles.sin().to(dtype=value.dtype).unsqueeze(1)

        even = value[..., 0::2]
        odd = value[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return value * torch.repeat_interleave(cos, 2, dim=-1) + rotated * torch.repeat_interleave(
            sin, 2, dim=-1
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query, key, value = self.c_attn(hidden_states).split(
            self.n_head * self.head_dim, dim=-1
        )
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if self.position_embedding == "rope":
            query = self._apply_rope(query, position_ids)
            key = self._apply_rope(key, position_ids)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        if attention_mask is None:
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            # Boolean SDPA masks use True for entries that are allowed.
            causal = torch.ones(
                seq_len, seq_len, device=hidden_states.device, dtype=torch.bool
            ).tril()
            allowed = causal.view(1, 1, seq_len, seq_len)
            allowed = allowed & attention_mask.to(torch.bool).view(batch_size, 1, 1, seq_len)
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed,
                dropout_p=dropout_p,
                is_causal=False,
            )

        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_head * self.head_dim
        )
        output = self.c_proj(context)
        return self.resid_dropout(output)


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2LMConfig):
        super().__init__()
        inner_dim = config.n_inner or (4 * config.n_embd)
        self.c_fc = nn.Linear(config.n_embd, inner_dim, bias=True)
        self.c_proj = nn.Linear(inner_dim, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.activation_function = config.activation_function

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        name = self.activation_function
        if name == "gelu_new":
            return 0.5 * x * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        if name == "gelu":
            return F.gelu(x)
        if name == "relu":
            return F.relu(x)
        if name == "silu":
            return F.silu(x)
        raise ValueError(f"Unsupported GPT-2 activation function: {name}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self._activation(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


class GPT2Block(nn.Module):
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
            if position_embedding == "ape"
            else None
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPT2Block(config, i, position_embedding) for i in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

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
        if self.wpe is not None:
            hidden_states = hidden_states + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        for block in self.h:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return self.ln_f(hidden_states)


class GPT2LMForCausalLM(nn.Module):
    """GPT-2 decoder with pretrained-checkpoint loading and optional RoPE."""

    def __init__(self, config: GPT2LMConfig, position_embedding: str = "ape"):
        super().__init__()
        if position_embedding not in {"ape", "rope"}:
            raise ValueError("position_embedding must be 'ape' or 'rope'")
        if position_embedding == "rope" and (config.n_embd // config.n_head) % 2:
            raise ValueError("RoPE requires an even head dimension")
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
                raise ValueError(
                    "labels must have the same shape as input_ids; the data loader "
                    "already shifts labels to the next token"
                )
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
        position_embedding: str = "ape",
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> tuple["GPT2LMForCausalLM", dict[str, Any]]:
        """Load Hugging Face GPT-2 weights, transposing Conv1D matrices.

        For ``position_embedding='rope'``, the pretrained ``wpe`` parameter is
        intentionally omitted. All token, attention, MLP, and LayerNorm weights
        are loaded from the GPT-2 checkpoint.
        """
        try:
            from transformers import GPT2LMHeadModel
        except ImportError as exc:  # pragma: no cover - depends on runtime setup
            raise ImportError(
                "Loading pretrained GPT-2 requires transformers. Install the "
                "repository requirements plus datasets."
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
        transpose_fragments = (
            ".attn.c_attn.weight",
            ".attn.c_proj.weight",
            ".mlp.c_fc.weight",
            ".mlp.c_proj.weight",
        )
        missing: list[str] = []
        for name, target_tensor in target_state.items():
            source_tensor = source_state.get(name)
            if source_tensor is None:
                missing.append(name)
                continue
            if any(fragment in name for fragment in transpose_fragments):
                source_tensor = source_tensor.transpose(0, 1).contiguous()
            if source_tensor.shape != target_tensor.shape:
                raise RuntimeError(
                    f"Checkpoint shape mismatch for {name}: source "
                    f"{tuple(source_tensor.shape)}, target {tuple(target_tensor.shape)}"
                )
            converted[name] = source_tensor

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
            "position_embedding": position_embedding,
            "loaded_tensor_count": len(converted),
            "dropped_absolute_position_embedding": position_embedding == "rope",
            "pretrained_weights_loaded": True,
        }
        return model, report

    def config_dict(self) -> dict[str, Any]:
        result = asdict(self.config)
        result["position_embedding"] = self.position_embedding
        return result
