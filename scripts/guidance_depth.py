"""
Depth guidance for Stage 1 of SAM 3D.

Uses a target depth map instead of a silhouette mask. Open3DHOI already contains ZoeDepth per pixel estimates

Plug into the denoising loop in inference_pipeline.py:

    from guidance_depth import DepthGuidance

    guidance = DepthGuidance(
        "path/to/depth.npy",
        guidance_scale=5.0,
        device="cuda",
    )

    # inside sample_sparse_structure(), pass guidance=guidance
"""

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


def _load_gt_depth(depth_path: str, size: int = 256) -> torch.Tensor:
    """
    Loads a depth npy and returns float tensor in [0,1].

    Assumes:
    - background = 0
    - foreground depth > 0
    """

    depth = np.load(depth_path).astype(np.float32)

    if depth.ndim == 3:
        depth = depth[..., 0]

    depth = torch.from_numpy(depth).float()

    # resize while preserving values (IMPORTANT)
    depth = torch.nn.functional.interpolate(
        depth[None, None, ...],
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return depth


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


def _render_depth(
    verts: torch.Tensor,
    faces: torch.Tensor,
    pose_rotation: torch.Tensor,
    pose_translation: torch.Tensor,
    pose_scale: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns differentiable depth map.

    Output:
        depth: (H, W)

    Background pixels are -1.
    """

    scale = pose_scale.float().to(device).mean()

    scaled_verts = verts.to(device) * scale

    mesh = Meshes(
        verts=[scaled_verts],
        faces=[faces.long().to(device)],
    )

    quat = pose_rotation.reshape(1, 4).to(device)

    R = quaternion_to_matrix(quat)

    T = pose_translation.reshape(1, 3).to(device)

    K = intrinsics.to(device)

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
        device=device,
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

    fragments = rasterizer(mesh)

    depth = fragments.zbuf[0, ..., 0]

    return depth


def _depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
) -> torch.Tensor:
    """
    Scale-invariant normalized depth loss.
    """

    gt = gt_depth.to(pred_depth.device)

    valid = (gt > 0) & (pred_depth > 0)

    if valid.sum() == 0:
        return torch.tensor(
            0.0,
            device=pred_depth.device,
            requires_grad=True,
        )

    pred = pred_depth[valid]
    target = gt[valid]

    # normalize for scale invariance
    pred = (pred - pred.mean()) / (pred.std() + 1e-8)

    target = (target - target.mean()) / (target.std() + 1e-8)

    loss = ((pred - target) ** 2).mean()

    return loss


class DepthGuidance:
    """
    Differentiable depth-map guidance for SAM3D sparse structure diffusion.

    At each denoising step:

        ss_grid
            -> FlexiCubes mesh
            -> differentiable depth render
            -> compare to GT depth
            -> backprop into ss_grid

    Usage:

        guidance = DepthGuidance(
            "depth.png",
            guidance_scale=5.0,
            device="cuda",
        )

        ss_step = guidance.apply(
            ss_step,
            pose_step["rotation"],
            pose_step["translation"],
            pose_step["scale"],
            intrinsics,
            float(t_step),
        )
    """

    def __init__(
        self,
        depth_path: str,
        guidance_scale: float = 5.0,
        start_t: float = 0.5,
        image_size: int = 256,
        device: str = "cpu",
    ):

        self.gt_depth = _load_gt_depth(
            depth_path,
            image_size,
        )

        self.scale = guidance_scale

        self.start_t = start_t

        self.image_size = image_size

        self.device = torch.device(device)

    @torch.enable_grad()
    def apply(
        self,
        ss_grid: torch.Tensor,          # (1,C,64,64,64)
        pose_rotation: torch.Tensor,   # quaternion
        pose_translation: torch.Tensor,
        pose_scale: torch.Tensor,
        intrinsics: torch.Tensor,
        t: float,
    ) -> torch.Tensor:

        if t < self.start_t:
            return ss_grid

        N = ss_grid.shape[-1]

        # fp32 required
        grid = (
            ss_grid.detach()
            .float()
            .to(self.device)
            .requires_grad_(True)
        )

        # FlexiCubes expects negative inside
        scalar_field = -grid[0, 0].reshape(-1)

        # Empty grid
        if (scalar_field < 0).sum() == 0:
            return ss_grid

        from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.flexicubes.flexicubes import (
            FlexiCubes,
        )

        voxelgrid_vertices, cube_idx = _build_flexicubes_grid(
            N,
            self.device,
        )

        fc = FlexiCubes(device=self.device)

        verts, faces, _, _ = fc(
            voxelgrid_vertices,
            scalar_field,
            cube_idx,
            resolution=N - 1,
        )

        if verts.shape[0] == 0:
            return ss_grid

        pred_depth = _render_depth(
            verts,
            faces,
            pose_rotation,
            pose_translation,
            pose_scale,
            intrinsics,
            self.image_size,
            self.device,
        )

        loss = _depth_loss(
            pred_depth,
            self.gt_depth,
        )

        loss.backward()

        with torch.no_grad():

            corrected = (
                ss_grid
                - self.scale
                * grid.grad.to(ss_grid.device)
            )

        print(
            f"  [depth guidance]"
            f" t={t:.3f}"
            f" loss={loss.item():.6f}"
            f" grad_norm={grid.grad.norm().item():.6f}"
        )

        return corrected