"""
Silhouette guidance for Stage 1 of SAM 3D.

Plug into the denoising loop in inference_pipeline.py:

    from guidance import SilhouetteGuidance
    guidance = SilhouetteGuidance("path/to/mask.png", device="cuda")
    # inside sample_sparse_structure(), pass guidance=guidance
"""

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix


def _load_gt_mask(mask_path: str, size: int = 256) -> torch.Tensor:
    arr = np.array(Image.open(mask_path))
    mask = arr > 0
    if mask.ndim == 3:
        mask = mask[..., -1]  # alpha channel
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (size, size), Image.NEAREST
    )
    return torch.from_numpy(np.array(resized) > 0).float()


def _build_flexicubes_grid(
    N: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.arange(N, device=device, dtype=torch.float) / N - 0.5
    gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")
    voxelgrid_vertices = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    flat_idx = torch.arange(N * N * N, device=device).reshape(N, N, N)
    i, j, k = torch.meshgrid(
        torch.arange(N - 1, device=device),
        torch.arange(N - 1, device=device),
        torch.arange(N - 1, device=device),
        indexing="ij",
    )
    i, j, k = i.reshape(-1), j.reshape(-1), k.reshape(-1)
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


def _render_soft_silhouette(
    verts: torch.Tensor,
    faces: torch.Tensor,
    pose_rotation: torch.Tensor,
    pose_translation: torch.Tensor,
    pose_scale: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Returns (H, W) soft alpha in [0, 1], differentiable w.r.t. verts."""
    scale = pose_scale.float().to(device).mean()
    scaled_verts = verts.to(device) * scale

    mesh = Meshes(
        verts=scaled_verts.unsqueeze(0),
        faces=faces.to(device).long().unsqueeze(0),
    )

    quat = pose_rotation.reshape(1, 4).to(device)
    R = quaternion_to_matrix(quat)  # (1, 3, 3)
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

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=50,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    output = renderer(mesh)  # (1, H, W, 4)
    return output[0, ..., 3]  # (H, W)


def _soft_iou_loss(pred_alpha: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """Soft IoU loss in [0, 1]. 0 = perfect overlap."""
    gt = gt_mask.to(pred_alpha.device)
    intersection = (pred_alpha * gt).sum()
    union = (pred_alpha + gt - pred_alpha * gt).sum()
    return 1.0 - intersection / (union + 1e-8)


class SilhouetteGuidance:
    """
    At each denoising step, takes the predicted clean ss_grid, renders a soft
    silhouette, computes soft IoU loss against a GT mask, and returns a
    gradient-corrected ss_grid.

    Usage in inference_pipeline.py::sample_sparse_structure():

        # after torch.save(...)
        if guidance is not None and intrinsics is not None:
            ss_step = guidance.apply(
                ss_step,
                pose_step["rotation"],
                pose_step["translation"],
                pose_step["scale"],
                intrinsics,
                float(t_step),
            )
            coords_step = torch.argwhere(ss_step > 0)[:, [0, 2, 3, 4]].int()
    """

    def __init__(
        self,
        mask_path: str,
        guidance_scale: float = 5.0,
        start_t: float = 0.5,
        image_size: int = 256,
        device: str = "cpu",
    ):
        self.gt_mask = _load_gt_mask(mask_path, image_size)
        self.scale = guidance_scale
        self.start_t = start_t
        self.image_size = image_size
        self.device = torch.device(device)

    @torch.enable_grad()
    def apply(
        self,
        ss_grid: torch.Tensor,          # (1, C, 64, 64, 64) predicted clean grid
        pose_rotation: torch.Tensor,    # (4,)  quaternion w,x,y,z
        pose_translation: torch.Tensor, # (3,)
        pose_scale: torch.Tensor,       # (3,) or scalar
        intrinsics: torch.Tensor,       # (3, 3) normalised camera matrix
        t: float,                       # current timestep in [0, 1]
    ) -> torch.Tensor:
        """Returns corrected ss_grid. If t < start_t, returns ss_grid unchanged."""
        if t < self.start_t:
            return ss_grid

        N = ss_grid.shape[-1]  # 64

        # Float32 detach — FlexiCubes + pytorch3d need fp32; autocast may be active
        grid = ss_grid.detach().float().to(self.device).requires_grad_(True)

        # Positive inside → negate for FlexiCubes (expects negative inside)
        scalar_field = -grid[0, 0].reshape(-1)

        if (scalar_field < 0).sum() == 0:
            return ss_grid  # grid still empty, nothing to guide

        from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.flexicubes.flexicubes import FlexiCubes

        voxelgrid_vertices, cube_idx = _build_flexicubes_grid(N, self.device)
        fc = FlexiCubes(device=self.device)
        verts, faces, _, _ = fc(
            voxelgrid_vertices, scalar_field, cube_idx, resolution=N - 1
        )

        if verts.shape[0] == 0:
            return ss_grid

        pred_alpha = _render_soft_silhouette(
            verts,
            faces,
            pose_rotation,
            pose_translation,
            pose_scale,
            intrinsics,
            self.image_size,
            self.device,
        )
        loss = _soft_iou_loss(pred_alpha, self.gt_mask)
        loss.backward()

        with torch.no_grad():
            corrected = ss_grid - self.scale * grid.grad.to(ss_grid.device)

        print(
            f"  [guidance] t={t:.3f}  loss={loss.item():.4f}"
            f"  grad_norm={grid.grad.norm().item():.6f}"
        )
        return corrected
