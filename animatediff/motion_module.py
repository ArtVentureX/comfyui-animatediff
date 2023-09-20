import os
import torch
from torch import Tensor, nn

import math
from einops import rearrange, repeat

from comfy.ldm.modules.attention import FeedForward, CrossAttention


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


# Merge from https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
def get_encoding_max_len(mm_state_dict: dict[str, Tensor]) -> int:
    # use pos_encoder.pe entries to determine max length - [1, {max_length}, {320|640|1280}]
    for key in mm_state_dict.keys():
        if key.endswith("pos_encoder.pe"):
            return mm_state_dict[key].size(1)  # get middle dim
    raise ValueError(f"No pos_encoder.pe found in mm_state_dict")


def has_mid_block(mm_state_dict: dict[str, Tensor]):
    # check if keys contain mid_block
    for key in mm_state_dict.keys():
        if key.startswith("mid_block."):
            return True
    return False


class MotionWrapper(nn.Module):
    def __init__(self, mm_type: str, encoding_max_len: int = 24, is_v2=False):
        super().__init__()
        self.mm_type = mm_type
        self.is_v2 = is_v2

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.mid_block = None
        self.encoding_max_len = encoding_max_len

        for c in (320, 640, 1280, 1280):
            self.down_blocks.append(
                MotionModule(c, BlockType.DOWN, encoding_max_len=encoding_max_len)
            )
        for c in (1280, 1280, 640, 320):
            self.up_blocks.append(
                MotionModule(c, BlockType.UP, encoding_max_len=encoding_max_len)
            )
        if is_v2:
            self.mid_block = MotionModule(
                1280, BlockType.MID, encoding_max_len=encoding_max_len
            )

    @classmethod
    def from_state_dict(cls, mm_state_dict: dict[str, Tensor], mm_type: str):
        encoding_max_len = get_encoding_max_len(mm_state_dict)
        is_v2 = has_mid_block(mm_state_dict)

        mm = cls(mm_type, encoding_max_len=encoding_max_len, is_v2=is_v2)
        mm.load_state_dict(mm_state_dict, strict=False)
        return mm

    def set_video_length(self, video_length: int):
        for block in self.down_blocks:
            block.set_video_length(video_length)
        for block in self.up_blocks:
            block.set_video_length(video_length)
        if self.mid_block is not None:
            self.mid_block.set_video_length(video_length)


class BlockType:
    UP = "up"
    DOWN = "down"
    MID = "mid"


class MotionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        block_type: BlockType,
        encoding_max_len=24,
    ):
        super().__init__()
        self.block_type = block_type

        if block_type == BlockType.MID:
            self.motion_modules = nn.ModuleList(
                [get_motion_module(in_channels, encoding_max_len)]
            )
        else:
            self.motion_modules = nn.ModuleList(
                [
                    get_motion_module(in_channels, encoding_max_len),
                    get_motion_module(in_channels, encoding_max_len),
                ]
            )
            if block_type == BlockType.UP:
                self.motion_modules.append(
                    get_motion_module(in_channels, encoding_max_len)
                )

    def set_video_length(self, video_length: int):
        for motion_module in self.motion_modules:
            motion_module.set_video_length(video_length)


def get_motion_module(in_channels, max_len):
    return VanillaTemporalModule(
        in_channels=in_channels, temporal_position_encoding_max_len=max_len
    )


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=1,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=True,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels
            // num_attention_heads
            // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(
                self.temporal_transformer.proj_out
            )

    def set_video_length(self, video_length: int):
        self.temporal_transformer.set_video_length(video_length)

    def forward(self, input_tensor, encoder_hidden_states, attention_mask=None):
        return self.temporal_transformer(
            input_tensor, encoder_hidden_states, attention_mask
        )


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.video_length = 16

    def set_video_length(self, video_length: int):
        self.video_length = video_length

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * weight, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=self.video_length,
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        output = hidden_states + residual

        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    context_dim=cross_attention_dim
                    if block_name.endswith("_Cross")
                    else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    # bias=attention_bias, # remove for Comfy CrossAttention
                    # upcast_attention=upcast_attention, # remove for Comfy CrossAttention
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, glu=(activation_fn == "geglu"))
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
    ):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if attention_block.is_cross_attention
                    else None,
                    video_length=video_length,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class VersatileAttention(CrossAttention):
    def __init__(
        self,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["context_dim"] is not None

        self.pos_encoder = (
            PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.0,
                max_len=temporal_position_encoding_max_len,
            )
            if (temporal_position_encoding and attention_mode == "Temporal")
            else None
        )

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        **cross_attention_kwargs,
    ):
        if self.attention_mode != "Temporal":
            raise NotImplementedError

        d = hidden_states.shape[1]
        hidden_states = rearrange(
            hidden_states, "(b f) d c -> (b d) f c", f=video_length
        )

        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        encoder_hidden_states = (
            repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
            if encoder_hidden_states is not None
            else encoder_hidden_states
        )

        hidden_states = super().forward(
            hidden_states,
            encoder_hidden_states,
            value=None,
            mask=attention_mask,
        )

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
