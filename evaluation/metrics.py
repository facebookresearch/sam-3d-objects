"""Chamfer distance and F-score metrics for mesh pairs."""

from __future__ import annotations

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
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


def chamfer(points_a: torch.Tensor, points_b: torch.Tensor) -> float:
    """Symmetric Chamfer distance — mean of *unsquared* L2 nearest-neighbor distances.

    chamfer = (mean_{a in A} min_{b in B} ||a - b||_2
            +  mean_{b in B} min_{a in A} ||a - b||_2) / 2

    PyTorch3D's ``knn_points`` returns squared L2 distances; we ``sqrt`` before
    averaging so the value range matches the SAM 3D paper §D.3.1 (their full model
    scores 0.0400 on SA-3DAO at this convention; the squared version is ~10× smaller
    and would not be comparable).

    Inputs are ``(N, 3)`` and ``(M, 3)`` float tensors on the same device; N and M
    may differ.
    """
    a = points_a.unsqueeze(0)
    b = points_b.unsqueeze(0)
    d_a_to_b = knn_points(a, b, K=1).dists.squeeze(-1).squeeze(0).clamp_min(0.0).sqrt()
    d_b_to_a = knn_points(b, a, K=1).dists.squeeze(-1).squeeze(0).clamp_min(0.0).sqrt()
    return float((d_a_to_b.mean() + d_b_to_a.mean()) / 2.0)


def f_score(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    thresholds: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05),
) -> dict[float, float]:
    """F1 score at each threshold τ in unsquared L2 distance units.

    For each τ:
        precision = fraction of A-points within τ of any B-point
        recall    = fraction of B-points within τ of any A-point
        f1        = 2 * P * R / (P + R)   (0 when P = R = 0)

    Returns ``{τ: f1}``. Conventionally A = prediction, B = ground truth; F1 itself
    is symmetric in (A, B), only P and R swap.
    """
    a = points_a.unsqueeze(0)
    b = points_b.unsqueeze(0)
    d_a = knn_points(a, b, K=1).dists.squeeze(-1).squeeze(0).clamp_min(0.0).sqrt()
    d_b = knn_points(b, a, K=1).dists.squeeze(-1).squeeze(0).clamp_min(0.0).sqrt()
    out: dict[float, float] = {}
    for tau in thresholds:
        precision = float((d_a < tau).float().mean())
        recall = float((d_b < tau).float().mean())
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        out[float(tau)] = f1
    return out


