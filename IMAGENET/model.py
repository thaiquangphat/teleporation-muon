
import collections.abc
import math
from typing import Callable, Optional, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from typing_extensions import Unpack
from transformers.models.vit.modeling_vit import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    ViTConfig,
    ViTPatchEmbeddings,
    ViTIntermediate,
    ViTOutput,
    ViTPooler,
    PreTrainedModel,
    GradientCheckpointingLayer,
    ALL_ATTENTION_FUNCTIONS,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
    ACT2FN,
    TransformersKwargs,
    auto_docstring,
    eager_attention_forward,
)


def create_sinusoidal_positions(n_pos, dim, device=None, dtype=torch.float32):
    position_enc = torch.zeros((n_pos, dim), dtype=dtype, device=device)
    for pos in range(n_pos):
        for j in range(dim):
            position_enc[pos, j] = pos / math.pow(10000, 2 * (j // 2) / dim)

    sentinel = dim // 2 + dim % 2
    out = torch.zeros_like(position_enc)
    out[:, 0:sentinel] = torch.sin(position_enc[:, 0::2])
    out[:, sentinel:] = torch.cos(position_enc[:, 1::2])
    return out
class ViTRoPEEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig, use_mask_token: bool = False):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = self.dropout(embeddings)
        return embeddings


class ViTRoPESelfAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    def forward(
        self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None, sinusoidal_pos = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size
        # key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        # value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        # query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)  
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        if sinusoidal_pos is not None:
            query_layer, key_layer = self.apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        return context_layer, attention_probs
    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer

class ViTRoPESelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTRoPEAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTRoPESelfAttention(config)
        self.output = ViTRoPESelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: set[int]):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None, sinusoidal_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states, head_mask, sinusoidal_pos=sinusoidal_pos)
        output = self.output(self_attn_output, hidden_states)
        return output
class ViTRoPELayer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the timm implementation."""
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTRoPEAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None, sinusoidal_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm, head_mask, sinusoidal_pos=sinusoidal_pos)
        # first residual connection
        hidden_states = attention_output + hidden_states
        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        return layer_output


class ViTRoPEEncoder(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        num_patches = (config.image_size // config.patch_size)**2
        self.embed_positions = create_sinusoidal_positions(num_patches + 1, self.config.hidden_size // config.num_attention_heads)
        self.layer = nn.ModuleList([ViTRoPELayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None) -> BaseModelOutput:
        sinusoidal_pos = self.embed_positions[: hidden_states.size(1), :]
        sinusoidal_pos = sinusoidal_pos.to(hidden_states.device, hidden_states.dtype)     
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(hidden_states, layer_head_mask, sinusoidal_pos=sinusoidal_pos)
        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class ViTRoPEPreTrainedModel(PreTrainedModel):
    config: ViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViTRoPEEmbeddings", "ViTRoPELayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": ViTRoPELayer, "attentions": ViTRoPESelfAttention,}
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTRoPEEmbeddings):
            # module.position_embeddings.data = nn.init.trunc_normal_(
            #     module.position_embeddings.data.to(torch.float32),
            #     mean=0.0,
            #     std=self.config.initializer_range,
            # ).to(module.position_embeddings.dtype)
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

            if module.mask_token is not None:
                module.mask_token.data.zero_()


@auto_docstring
class ViTRoPEModel(ViTRoPEPreTrainedModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked image modeling.
        """
        super().__init__(config)
        self.config = config
        self.embeddings = ViTRoPEEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTRoPEEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None
        # Initialize weights and apply final processing
        self.post_init()
    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings
    def _prune_heads(self, heads_to_prune: dict[int, list[int]]):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs: BaseModelOutput = self.encoder(embedding_output, head_mask=head_mask)

        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output)


class ViTRoPEForImageClassification(ViTRoPEPreTrainedModel):
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vit = ViTRoPEModel(config, add_pooling_layer=False)
        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPooling = self.vit(
            pixel_values,
            head_mask=head_mask,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

