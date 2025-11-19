# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from torch import nn
from ..modules.transformer import (
    AbsolutePositionEmbedder,
)
from ..modules import spatial
import torch.nn.functional as F
from typing import Callable


class Latent(nn.Module):
    def __init__(
        self, in_channels, model_channels: int, pos_embedder: Callable[[], torch.Tensor]
    ):
        super().__init__()
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.out_layer = nn.Linear(model_channels, in_channels)

        pos_emb = pos_embedder()
        if isinstance(pos_emb, torch.nn.Parameter):
            # learnt position embedding
            self.register_parameter("pos_emb", pos_emb)
        elif isinstance(pos_emb, torch.Tensor):
            # fixed position embedding
            self.register_buffer("pos_emb", pos_emb)
        else:
            raise NotImplementedError

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out output layers:
        # nn.init.constant_(self.out_layer.weight, 0)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.constant_(self.out_layer.bias, 0)

    def to_input(self, x):
        x = self.input_layer(x)
        x = x + self.pos_emb[None]

        return x

    def to_output(self, h):
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        return h


def ShapePositionEmbedder(model_channels, resolution, patch_size):
    def embedder():
        pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
        coords = torch.meshgrid(
            *[torch.arange(res) for res in [resolution // patch_size] * 3],
            indexing="ij",
        )
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        pos_emb = pos_embedder(coords)

        return pos_emb

    return embedder


def RandomPositionEmbedder(model_channels, token_len):
    def embedder():
        pos_emb = torch.randn(token_len, model_channels)

        return pos_emb

    return embedder


def LearntPositionEmbedder(model_channels, token_len):
    def embedder():
        pos_emb = torch.nn.Parameter(torch.randn(token_len, model_channels))

        return pos_emb

    return embedder
