"""Voxelize a point cloud into a boolean occupancy grid for voxel-IoU."""

from __future__ import annotations

import torch


def voxelize_points(
    points: torch.Tensor,
    resolution: int = 64,
    extent: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Voxelize a point cloud into an ``(R, R, R)`` boolean occupancy grid.

    Each voxel covers an equal cell of the cubic ``extent`` box; a voxel is True
    iff it contains at least one input point. Points outside ``extent`` are
    silently dropped.

    This is *surface* voxelization — the input is a point cloud sampled from the
    mesh surface (e.g. via ``sample_points``), so only voxels the surface passes
    through are marked. Matches SAM 3D paper §D.3.1's voxel-IoU protocol.
    """
    lo, hi = extent
    span = hi - lo
    idx = ((points - lo) / span * resolution).floor().long()
    in_bounds = ((idx >= 0) & (idx < resolution)).all(dim=1)
    idx = idx[in_bounds]
    grid = torch.zeros(
        (resolution, resolution, resolution), dtype=torch.bool, device=points.device
    )
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid
