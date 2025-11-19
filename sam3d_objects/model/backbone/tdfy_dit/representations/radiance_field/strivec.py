# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..octree import DfsOctree as Octree


class Strivec(Octree):
    def __init__(
        self,
        resolution: int,
        aabb: list,
        sh_degree: int = 0,
        rank: int = 8,
        dim: int = 8,
        device: str = "cuda",
    ):
        assert np.log2(resolution) % 1 == 0, "Resolution must be a power of 2"
        self.resolution = resolution
        depth = int(np.round(np.log2(resolution)))
        super().__init__(
            depth=depth,
            aabb=aabb,
            sh_degree=sh_degree,
            primitive="trivec",
            primitive_config={"rank": rank, "dim": dim},
            device=device,
        )
