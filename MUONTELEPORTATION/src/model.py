from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """Kept for regression tests of the original two-layer teleportation helper."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.hidden(pixel_values.view(pixel_values.shape[0], -1)))
        return self.output(hidden)


@dataclass(frozen=True)
class ViTConfig:
    image_size: int
    num_channels: int
    num_classes: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    patch_size: int
    position_embedding: str = "rope"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12

    def __post_init__(self) -> None:
        if min(
            self.image_size,
            self.num_channels,
            self.num_classes,
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.intermediate_size,
            self.patch_size,
        ) < 1:
            raise ValueError("ViT dimensions must all be positive.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size.")
        if self.position_embedding not in {"rope", "learnable"}:
            raise ValueError("position_embedding must be 'rope' or 'learnable'.")


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.image_size = config.image_size
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape (batch, channels, height, width).")
        if pixel_values.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} image channels, received {pixel_values.shape[1]}."
            )
        if pixel_values.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} images, "
                f"received {pixel_values.shape[-2]}x{pixel_values.shape[-1]}."
            )
        patches = self.projection(pixel_values)
        return patches.flatten(2).transpose(1, 2)


class ViTEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = (
            nn.Parameter(torch.zeros(1, self.patch_embeddings.num_patches + 1, config.hidden_size))
            if config.position_embedding == "learnable"
            else None
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(pixel_values.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        return self.dropout(embeddings)


class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size
        self.position_embedding = config.position_embedding
        self.scaling = self.attention_head_size**-0.5
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

    def _split_heads(self, projected: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = projected.shape
        return projected.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(hidden_states))
        key = self._split_heads(self.key(hidden_states))
        value = self._split_heads(self.value(hidden_states))
        if self.position_embedding == "rope":
            query, key = self._apply_rotary_position_embedding(query, key)

        attention_scores = (query @ key.transpose(-2, -1)) * self.scaling
        attention_probs = self.dropout(attention_scores.softmax(dim=-1))
        context = attention_probs @ value
        context = context.transpose(1, 2).contiguous()
        return context.view(hidden_states.shape[0], hidden_states.shape[1], self.all_head_size)

    def _apply_rotary_position_embedding(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attention_head_size % 2:
            raise ValueError("RoPE requires an even attention head size.")
        positions = torch.arange(query.shape[-2], device=query.device, dtype=torch.float32)
        frequencies = 1.0 / (
            10000
            ** (
                torch.arange(
                    0,
                    self.attention_head_size,
                    2,
                    device=query.device,
                    dtype=torch.float32,
                )
                / self.attention_head_size
            )
        )
        angles = torch.outer(positions, frequencies).to(dtype=query.dtype)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]

        def rotate(tensor: torch.Tensor) -> torch.Tensor:
            even = tensor[..., 0::2]
            odd = tensor[..., 1::2]
            return torch.stack(
                (even * cos - odd * sin, even * sin + odd * cos),
                dim=-1,
            ).flatten(-2)

        return rotate(query), rotate(key)


class ViTSelfOutput(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states))


class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(hidden_states))


class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states)) + residual


class ViTLayer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.layernorm_before(hidden_states)
        hidden_states = self.attention(normalized) + hidden_states
        normalized = self.layernorm_after(hidden_states)
        return self.output(self.intermediate(normalized), hidden_states)


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [ViTLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


class ViTModel(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        return self.layernorm(hidden_states)[:, 0]


class ViTForImageClassification(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.num_labels = config.num_classes
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.apply(self._initialize_weights)
        nn.init.trunc_normal_(self.vit.embeddings.cls_token, std=0.02)
        if self.vit.embeddings.position_embeddings is not None:
            nn.init.trunc_normal_(self.vit.embeddings.position_embeddings, std=0.02)

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.classifier(self.vit(pixel_values)))


def build_vit(
    image_size: int,
    num_channels: int,
    num_classes: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    patch_size: int,
    position_embedding: str = "rope",
) -> ViTForImageClassification:
    config = ViTConfig(
        image_size=image_size,
        num_channels=num_channels,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        patch_size=patch_size,
        position_embedding=position_embedding,
    )
    return ViTForImageClassification(config)
