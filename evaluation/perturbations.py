"""Controlled mesh perturbations (translation, vertex jitter) for the synthetic test."""

from __future__ import annotations

import numpy as np


def translate(verts: np.ndarray, t: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """Translate vertices by ``t`` (a 3-vector). Returns a new ``(N, 3)`` array."""
    return verts + np.asarray(t, dtype=verts.dtype).reshape(1, 3)


def add_vertex_noise(verts: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Add isotropic Gaussian noise with std ``sigma`` to each vertex independently.

    ``sigma`` is in the same units as ``verts``; for meshes normalized to [-1, 1],
    σ = 0.01 is ~0.5% of the cube extent. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=verts.shape)
    return verts + noise.astype(verts.dtype)
