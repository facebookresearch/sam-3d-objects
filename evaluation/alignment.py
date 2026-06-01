"""ICP alignment for predicted-vs-ground-truth point clouds.

Two coarse-init modes are exposed via ``align_icp(..., mode=...)``:

  * ``"fixed"`` — apply ``fixed_rotation`` to the prediction, then ICP. Defaults to
    ``FIXED_ROTATION_DEFAULT``, the rotation that aligned the coffee_cup validation
    pair best (validated both by post-ICP chamfer and visually in Blender).
  * ``"grid"`` — try all 24 axis-aligned cube rotations as ICP initialisations and
    return the result with the lowest post-ICP chamfer. Robust to unknown axis
    conventions; costs 24× one ICP run per pair.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from pytorch3d.ops import iterative_closest_point, knn_points


# Row-vector convention: rotated = points @ FIXED_ROTATION_DEFAULT.T
# Mapping: new_x = -old_y, new_y = -old_z, new_z = +old_x  (idx 12 in cube_rotations()).
# Source: coarse_align_search on coffee_cup pair, confirmed visually in Blender.
FIXED_ROTATION_DEFAULT: torch.Tensor = torch.tensor(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)


def cube_rotations() -> list[np.ndarray]:
    """The 24 proper rotations of a cube (signed permutation matrices with det = +1)."""
    rots: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            R = np.zeros((3, 3))
            for i, p in enumerate(perm):
                R[i, p] = signs[i]
            if np.linalg.det(R) > 0.5:
                rots.append(R)
    return rots


def _run_icp(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    *,
    estimate_scale: bool,
    max_iterations: int,
) -> torch.Tensor:
    result = iterative_closest_point(
        points_pred.unsqueeze(0),
        points_gt.unsqueeze(0),
        max_iterations=max_iterations,
        estimate_scale=estimate_scale,
    )
    return result.Xt[0]


def _chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    # Local copy to avoid an import cycle with evaluation.metrics; symmetric mean of
    # unsquared L2 nearest-neighbour distances, same as metrics.chamfer.
    a4 = a.unsqueeze(0)
    b4 = b.unsqueeze(0)
    d_ab = knn_points(a4, b4, K=1).dists.squeeze(-1).squeeze(0).clamp_min(0.0).sqrt()
    d_ba = knn_points(b4, a4, K=1).dists.squeeze(-1).squeeze(0).clamp_min(0.0).sqrt()
    return float((d_ab.mean() + d_ba.mean()) / 2.0)


def align_icp(
    points_pred: torch.Tensor,
    points_gt: torch.Tensor,
    *,
    mode: str = "grid",
    fixed_rotation: torch.Tensor | None = None,
    estimate_scale: bool = False,
    max_iterations: int = 100,
    early_exit: bool = True,
) -> torch.Tensor:
    """Align ``points_pred`` to ``points_gt`` and return the transformed prediction.

    SAM 3D paper §D.3.1 applies point-to-point ICP to every (predicted, GT) pair
    after independent normalization to [-1, 1]. We additionally choose a coarse
    rotational init because the SAM 3D output and our GT meshes do not share an
    axis convention; vanilla ICP from identity falls into the wrong basin.

    ``mode="fixed"`` rotates by ``fixed_rotation`` (default ``FIXED_ROTATION_DEFAULT``)
    then runs ICP — cheap, requires that the input pair share the convention the
    rotation was calibrated for.

    ``mode="grid"`` runs ICP from each of the 24 cube rotations and returns the
    aligned cloud with the lowest post-ICP chamfer — robust but 24× more compute.
    When ``early_exit=True`` (default), the search stops as soon as a rotation
    achieves chamfer < 0.1 on the [-1, 1]-normalised scale, since further search
    is unlikely to improve on an already-good alignment.

    Both inputs are ``(N, 3)`` float tensors on the same device; returns ``(N, 3)``.
    """
    if mode == "fixed":
        rotation = FIXED_ROTATION_DEFAULT if fixed_rotation is None else fixed_rotation
        rotation = rotation.to(device=points_pred.device, dtype=points_pred.dtype)
        rotated = points_pred @ rotation.T
        return _run_icp(
            rotated, points_gt,
            estimate_scale=estimate_scale, max_iterations=max_iterations,
        )

    if mode == "grid":
        best_pts: torch.Tensor | None = None
        best_cd = float("inf")
        for R in cube_rotations():
            R_t = torch.as_tensor(R, dtype=points_pred.dtype, device=points_pred.device)
            rotated = points_pred @ R_t.T
            aligned = _run_icp(
                rotated, points_gt,
                estimate_scale=estimate_scale, max_iterations=max_iterations,
            )
            cd = _chamfer(aligned, points_gt)
            if cd < best_cd:
                best_cd = cd
                best_pts = aligned
            if early_exit and best_cd < 0.1:
                break
        assert best_pts is not None  # cube_rotations() is non-empty
        return best_pts

    raise ValueError(f"unknown mode {mode!r}; expected 'fixed' or 'grid'")
