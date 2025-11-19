# Copyright (c) Meta Platforms, Inc. and affiliates.
from functools import partial
from typing import *
from torch.utils import _pytree
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention
from sam3d_objects.data.utils import (
    tree_reduce_unique,
)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x, dim=-1) * self.gamma * self.scale).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**self.freqs)

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = (
            torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        )
        return x_embed

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (sp.SparseTensor): [..., N, D] tensor of queries
            k (sp.SparseTensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))

        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat(
                [
                    phases,
                    torch.polar(
                        torch.ones(
                            *phases.shape[:-1],
                            self.hidden_size // 2 - phases.shape[1],
                            device=phases.device,
                        ),
                        torch.zeros(
                            *phases.shape[:-1],
                            self.hidden_size // 2 - phases.shape[1],
                            device=phases.device,
                        ),
                    ),
                ],
                dim=-1,
            )
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert (
            type == "self" or attn_mode == "full"
        ), "Cross-attention only supports full attention"

        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h


class MOTMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        latent_names: List = None,
        protect_modality_list: List = ["shape"],
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert (
            type == "self" or attn_mode == "full"
        ), "Cross-attention only supports full attention"

        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        self.protect_modality_list = protect_modality_list

        if self._type == "self":
            self.to_qkv = torch.nn.ModuleDict(
                {
                    latent_name: nn.Linear(channels, channels * 3, bias=qkv_bias)
                    for latent_name in latent_names
                }
            )
        else:
            self.to_q = torch.nn.ModuleDict(
                {
                    latent_name: nn.Linear(channels, channels, bias=qkv_bias)
                    for latent_name in latent_names
                }
            )
            self.to_kv = torch.nn.ModuleDict(
                {
                    latent_name: nn.Linear(
                        self.ctx_channels, channels * 2, bias=qkv_bias
                    )
                    for latent_name in latent_names
                }
            )

        if self.qk_rms_norm:
            self.q_rms_norm = torch.nn.ModuleDict(
                {
                    latent_name: MultiHeadRMSNorm(self.head_dim, num_heads)
                    for latent_name in latent_names
                }
            )
            self.k_rms_norm = torch.nn.ModuleDict(
                {
                    latent_name: MultiHeadRMSNorm(self.head_dim, num_heads)
                    for latent_name in latent_names
                }
            )

        self.to_out = torch.nn.ModuleDict(
            {latent_name: nn.Linear(channels, channels) for latent_name in latent_names}
        )

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    def _reshape(self, qkv, tensor_shape, num_heads):
        B, L, _ = tensor_shape
        return qkv.reshape(B, L, 3, num_heads, -1)

    def _reshape_back(self, qkv, tensor_shape):
        B, L, _ = tensor_shape
        return qkv.reshape(B, L, -1)

    def _apply_module(self, x, module):
        return module(x)

    # This is stupid, _pytree does not support ModuleDict
    def _moduledict_to_dict(self, module):
        return {key: module for key, module in module.items()}

    def unbind_qkv(self, qkv):
        q, k, v = {}, {}, {}
        for latent_name, _qkv in qkv.items():
            _q, _k, _v = _qkv.unbind(dim=2)
            q[latent_name] = _q
            k[latent_name] = _k
            v[latent_name] = _v

        return q, k, v

    def _get_shape(self, x):
        return x.shape

    def concatenate_tensor(self, tensor_dict, latent_names):
        merged = []
        indicies_mapping = {}
        total_tokens = 0
        for latent_name in latent_names:
            merged.append(tensor_dict[latent_name])
            cur_token_len = tensor_dict[latent_name].shape[1]
            indicies_mapping[latent_name] = [total_tokens, cur_token_len]
            total_tokens += cur_token_len
        # merge along token dimension
        return torch.cat(merged, dim=1), indicies_mapping

    def unpack_tensors(self, h_others, indicies_mapping):
        h = {}
        for latent_name, (start, cur_token_len) in indicies_mapping.items():
            h[latent_name] = h_others[:, start : start + cur_token_len]

        return h

    def mm_scale_dot_product_attention(self, q, k, v):
        h = {}
        latent_names = list(q.keys())
        # for protected modality, it only attends itself
        for protect_modality in self.protect_modality_list:
            _q = q[protect_modality]
            _k = k[protect_modality]
            _v = v[protect_modality]
            h[protect_modality] = scaled_dot_product_attention(_q, _k, _v)

        # for the rest it is ok to attend each other and allow gradient
        other_modalities = [
            n for n in latent_names if n not in self.protect_modality_list
        ]
        _q, indicies_mapping = self.concatenate_tensor(q, other_modalities)
        o_k, _ = self.concatenate_tensor(k, other_modalities)
        o_v, _ = self.concatenate_tensor(v, other_modalities)
        # no gradiant flow back to protected modality (e.g. shape)
        _k, _ = self.concatenate_tensor(k, self.protect_modality_list)
        _v, _ = self.concatenate_tensor(v, self.protect_modality_list)
        _k = _k.detach()
        _v = _v.detach()
        _k = torch.cat([o_k, _k], dim=1)
        _v = torch.cat([o_v, _v], dim=1)
        h_others = scaled_dot_product_attention(_q, _k, _v)
        h.update(self.unpack_tensors(h_others, indicies_mapping))

        return h

    def forward(
        self,
        x: Dict,
    ) -> torch.Tensor:
        shapes = _pytree.tree_map(self._get_shape, x)
        if self._type == "self":
            qkv = _pytree.tree_map(
                self._apply_module, x, self._moduledict_to_dict(self.to_qkv)
            )
            qkv = _pytree.tree_map(
                partial(self._reshape, num_heads=self.num_heads), qkv, shapes
            )
            if self.use_rope:
                raise NotImplementedError
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = self.unbind_qkv(qkv)
                    q = _pytree.tree_map(
                        self._apply_module, q, self._moduledict_to_dict(self.q_rms_norm)
                    )
                    k = _pytree.tree_map(
                        self._apply_module, k, self._moduledict_to_dict(self.k_rms_norm)
                    )
                    h = self.mm_scale_dot_product_attention(q, k, v)
                else:
                    raise NotImplementedError
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            raise NotImplementedError

        h = _pytree.tree_map(self._reshape_back, h, shapes)
        h = _pytree.tree_map(
            self._apply_module, h, self._moduledict_to_dict(self.to_out)
        )
        return h
