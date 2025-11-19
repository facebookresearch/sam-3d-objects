# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...modules import sparse as sp
from .base import SparseTransformerBase
from ...representations import Strivec


class SLatRadianceFieldDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self._calc_layout()
        self.out_layer = sp.SparseLinear(model_channels, self.out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _calc_layout(self) -> None:
        self.layout = {
            "trivec": {
                "shape": (self.rep_config["rank"], 3, self.rep_config["dim"]),
                "size": self.rep_config["rank"] * 3 * self.rep_config["dim"],
            },
            "density": {
                "shape": (self.rep_config["rank"],),
                "size": self.rep_config["rank"],
            },
            "features_dc": {
                "shape": (self.rep_config["rank"], 1, 3),
                "size": self.rep_config["rank"] * 3,
            },
        }
        start = 0
        for k, v in self.layout.items():
            v["range"] = (start, start + v["size"])
            start += v["size"]
        self.out_channels = start

    def to_representation(self, x: sp.SparseTensor) -> List[Strivec]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            representation = Strivec(
                sh_degree=0,
                resolution=self.resolution,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                rank=self.rep_config["rank"],
                dim=self.rep_config["dim"],
                device="cuda",
            )
            representation.density_shift = 0.0
            representation.position = (
                x.coords[x.layout[i]][:, 1:].float() + 0.5
            ) / self.resolution
            representation.depth = torch.full(
                (representation.position.shape[0], 1),
                int(np.log2(self.resolution)),
                dtype=torch.uint8,
                device="cuda",
            )
            for k, v in self.layout.items():
                setattr(
                    representation,
                    k,
                    x.feats[x.layout[i]][:, v["range"][0] : v["range"][1]].reshape(
                        -1, *v["shape"]
                    ),
                )
            representation.trivec = representation.trivec + 1
            ret.append(representation)
        return ret

    def forward(self, x: sp.SparseTensor) -> List[Strivec]:
        h = super().forward(x)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return self.to_representation(h)
