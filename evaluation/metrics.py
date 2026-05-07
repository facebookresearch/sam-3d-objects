"""Chamfer distance, F-score, voxel IoU, and EMD metrics for mesh pairs."""

from __future__ import annotations

import numpy as np
import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes


def sample_points(
    verts: np.ndarray,
    faces: np.ndarray,
    n: int = 1_000_000,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Uniformly sample ``n`` points from a mesh's surface (area-weighted).

    Returns a ``(n, 3)`` float32 tensor on ``device``. Determinism is controlled by
    ``seed``: the same (mesh, n, seed) triple always yields the same point cloud.
    PyTorch3D's ``sample_points_from_meshes`` reads from the global RNG, so we fork
    the RNG state to keep seeding hermetic.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    verts_t = torch.as_tensor(verts, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces, dtype=torch.int64, device=device)
    mesh = Meshes(verts=[verts_t], faces=[faces_t])

    cuda_devices = [device.index if device.index is not None else 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        points = sample_points_from_meshes(mesh, num_samples=n)
    return points[0]
