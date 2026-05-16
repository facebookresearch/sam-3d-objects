"""
Compare intermediate diffusion steps against a ground-truth depth map.

For each saved step_T.pt file:
  1. Extract a mesh from the voxel grid using FlexiCubes.
  2. Render a depth map using the saved camera pose/intrinsics.
  3. Compare against the GT depth map.
  4. Save a side-by-side visualisation.

Usage:
    uv run python scripts/compare_depth_steps.py \
        --steps-dir .cache/ss_steps \
        --depth depth.png \
        --output outputs/depth_comparison
"""

import glob
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
)

from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.flexicubes.flexicubes import (
    FlexiCubes,
)

# ============================================================
# Grid construction
# ============================================================


def _build_flexicubes_grid(
    N: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:

    xs = torch.arange(
        N,
        device=device,
        dtype=torch.float,
    ) / N - 0.5

    gx, gy, gz = torch.meshgrid(
        xs,
        xs,
        xs,
        indexing="ij",
    )

    voxelgrid_vertices = torch.stack(
        [gx, gy, gz],
        dim=-1,
    ).reshape(-1, 3)

    flat_idx = torch.arange(
        N * N * N,
        device=device,
    ).reshape(N, N, N)

    i, j, k = torch.meshgrid(
        torch.arange(N - 1, device=device),
        torch.arange(N - 1, device=device),
        torch.arange(N - 1, device=device),
        indexing="ij",
    )

    i = i.reshape(-1)
    j = j.reshape(-1)
    k = k.reshape(-1)

    cube_idx = torch.stack(
        [
            flat_idx[i, j, k],
            flat_idx[i + 1, j, k],
            flat_idx[i, j + 1, k],
            flat_idx[i + 1, j + 1, k],
            flat_idx[i, j, k + 1],
            flat_idx[i + 1, j, k + 1],
            flat_idx[i, j + 1, k + 1],
            flat_idx[i + 1, j + 1, k + 1],
        ],
        dim=-1,
    )

    return voxelgrid_vertices, cube_idx


# ============================================================
# Mesh extraction
# ============================================================


def extract_mesh(
    ss_grid: torch.Tensor,
    device: str = "cpu",
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:

    dev = torch.device(device)

    N = ss_grid.shape[-1]

    # FlexiCubes expects negative inside
    scalar_field = -ss_grid[0, 0].float().to(dev).reshape(-1)

    if (scalar_field < 0).sum() == 0:
        return None

    voxelgrid_vertices, cube_idx = _build_flexicubes_grid(
        N,
        dev,
    )

    fc = FlexiCubes(device=dev)

    with torch.no_grad():
        verts, faces, _, _ = fc(
            voxelgrid_vertices,
            scalar_field,
            cube_idx,
            resolution=N - 1,
        )

    if verts.shape[0] == 0:
        return None

    return verts, faces


# ============================================================
# Depth rendering
# ============================================================


def render_depth(
    verts: torch.Tensor,
    faces: torch.Tensor,
    step_data: dict,
    image_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """
    Render differentiable depth map using saved camera parameters.

    Returns:
        depth: (H,W) float32
            background = -1
    """

    dev = torch.device(device)

    scaled_verts = verts.to(dev)

    if "pose_scale" in step_data:
        scale = step_data["pose_scale"].float().to(dev).mean()
        scaled_verts = scaled_verts * scale

    mesh = Meshes(
        verts=[scaled_verts],
        faces=[faces.long().to(dev)],
    )

    quat = step_data["pose_rotation"].reshape(1, 4).to(dev)

    R = quaternion_to_matrix(quat)

    T = step_data["pose_translation"].reshape(1, 3).to(dev)

    K = step_data["intrinsics"].to(dev)

    fx_px = K[0, 0] * image_size
    fy_px = K[1, 1] * image_size
    cx_px = K[0, 2] * image_size
    cy_px = K[1, 2] * image_size

    cameras = PerspectiveCameras(
        focal_length=((fx_px, fy_px),),
        principal_point=((cx_px, cy_px),),
        R=R,
        T=T,
        in_ndc=False,
        image_size=((image_size, image_size),),
        device=dev,
    )

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    )

    with torch.no_grad():
        fragments = rasterizer(mesh)

    depth = fragments.zbuf[0, ..., 0]

    return depth.cpu().numpy()


# ============================================================
# GT depth loading
# ============================================================


def load_gt_depth(depth_path: str, size: int = 256) -> np.ndarray:

    depth = np.load(depth_path).astype(np.float32)

    if depth.ndim == 3:
        depth = depth[..., 0]

    depth = torch.from_numpy(depth).float()

    depth = torch.nn.functional.interpolate(
        depth[None, None],
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return depth.numpy()


# ============================================================
# Metrics
# ============================================================


def normalize_depth(depth: np.ndarray) -> np.ndarray:

    valid = depth > 0

    if valid.sum() == 0:
        return depth

    d = depth.copy()

    dmin = d[valid].min()
    dmax = d[valid].max()

    d[valid] = (d[valid] - dmin) / (dmax - dmin + 1e-8)

    return d


def compute_depth_mse(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
) -> float:

    valid = (pred_depth > 0) & (gt_depth > 0)

    if valid.sum() == 0:
        return float("inf")

    pred = pred_depth[valid]
    gt = gt_depth[valid]

    pred = (pred - pred.mean()) / (pred.std() + 1e-8)
    gt = (gt - gt.mean()) / (gt.std() + 1e-8)

    mse = ((pred - gt) ** 2).mean()

    return float(mse)


# ============================================================
# Visualisation
# ============================================================


def save_comparison(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    error_map: np.ndarray,
    t_step: float,
    mse: float,
    output_path: str,
) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    fig.suptitle(f"t={t_step:.3f}  depth_mse={mse:.6f}")

    axes[0].imshow(pred_depth, cmap="viridis")
    axes[0].set_title("Predicted depth")

    axes[1].imshow(gt_depth, cmap="viridis")
    axes[1].set_title("GT depth")

    axes[2].imshow(error_map, cmap="magma")
    axes[2].set_title("Absolute error")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(output_path, dpi=100)

    plt.close(fig)


# ============================================================
# Main
# ============================================================


def main(
    steps_dir: str,
    depth_path: str,
    output_dir: str,
    image_size: int = 256,
    device: str = "cpu",
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(
        glob.glob(
            os.path.join(steps_dir, "step_*.pt")
        )
    )

    if not pt_files:
        raise FileNotFoundError(
            f"No step_*.pt files found in {steps_dir}"
        )

    gt_depth = load_gt_depth(
        depth_path,
        size=image_size,
    )

    print("Loaded GT depth")

    for pt_path in pt_files:

        step_data = torch.load(
            pt_path,
            map_location="cpu",
            weights_only=False,
        )

        t_step = float(step_data["t_step"])

        print(f"t={t_step:.3f}", end="  ")

        mesh = extract_mesh(
            step_data["ss_grid"],
            device=device,
        )

        if mesh is None:
            print("no mesh")
            continue

        verts, faces = mesh

        pred_depth = render_depth(
            verts,
            faces,
            step_data,
            image_size=image_size,
            device=device,
        )

        # pred_depth = normalize_depth(pred_depth)

        mse = compute_depth_mse(
            pred_depth,
            gt_depth,
        )

        print(f"depth_mse={mse:.6f}")

        valid = (pred_depth > 0) & (gt_depth > 0)

        error_map = np.zeros_like(pred_depth)

        error_map[valid] = np.abs(
            pred_depth[valid] - gt_depth[valid]
        )

        out_path = os.path.join(
            output_dir,
            f"step_{t_step:.3f}.png",
        )

        save_comparison(
            pred_depth,
            gt_depth,
            error_map,
            t_step,
            mse,
            out_path,
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--steps-dir",
        default=".cache/ss_steps",
    )

    parser.add_argument(
        "--depth",
        required=True,
        dest="depth_path",
    )

    parser.add_argument(
        "--output",
        default="outputs/depth_comparison",
        dest="output_dir",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    main(
        steps_dir=args.steps_dir,
        depth_path=args.depth_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        device=args.device,
    )