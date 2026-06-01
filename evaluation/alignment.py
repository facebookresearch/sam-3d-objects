"""ICP alignment for predicted-vs-ground-truth point clouds."""

from __future__ import annotations

import torch
from pytorch3d.ops import iterative_closest_point


def align_icp(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    estimate_scale: bool = False,
    max_iterations: int = 100,
) -> torch.Tensor:
    """Align ``points_pred`` to ``points_gt`` via ICP and return the transformed prediction.

    SAM 3D paper §D.3.1 applies ICP to every (predicted, GT) pair after independent
    normalization to [-1, 1] and before metrics. Defaults to point-to-point ICP
    with rigid alignment (``estimate_scale=False``); the paper's normalization
    already removes scale ambiguity.

    Both inputs are ``(N, 3)`` float tensors on the same device; returns ``(N, 3)``.
    """
    src = points_pred.unsqueeze(0)
    tgt = points_gt.unsqueeze(0)
    result = iterative_closest_point(
        src,
        tgt,
        max_iterations=max_iterations,
        estimate_scale=estimate_scale,
    )
    return result.Xt[0]
