"""Load .obj / .ply / .glb meshes and normalize them to a unit bbox."""

from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a mesh from disk and return (vertices, faces) as numpy arrays.

    Vertices are float64 (N, 3); faces are int64 (M, 3). `.glb` scenes containing
    multiple primitives are flattened into a single mesh via `force='mesh'`.
    """
    mesh = trimesh.load(str(path), force="mesh", process=False)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def normalize_mesh(verts: np.ndarray) -> np.ndarray:
    """Center the axis-aligned bbox at the origin and scale so the longest dim equals 2.0.

    Maps the mesh into the [-1, 1] cube — the convention used by SAM 3D paper §D.3.1
    and assumed by F-score thresholds like F1@0.01. Must be applied to both meshes
    independently before alignment / metrics.
    """
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    # Assumes the mesh has nonzero extent on at least one axis. A degenerate
    # single-point mesh would divide by zero here; we don't guard because real
    # loaded meshes always have extent and a silent inf is easier to debug than
    # a wrong-but-finite answer from a fallback.
    scale = 2.0 / (bbox_max - bbox_min).max()
    return (verts - center) * scale


def save_mesh(verts: np.ndarray, faces: np.ndarray, path: str | Path) -> None:
    """Write a triangle mesh to disk. Format is inferred from the path's extension."""
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(str(path))
