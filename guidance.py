"""
Guidance classes for Stage 1 of SAM 3D.

HOW INJECTION WORKS
-------------------
solve_iter() yields x_t. Calling x_t[key].data.copy_() before the next
iteration modifies the exact tensor the ODE solver reads on its next step.

ARCHITECTURE
------------
BaseGuidance          abstract base — apply() returns dict of corrected x_t keys
  ShapeGuidance       soft-IoU loss          → x_t["shape"]
  PoseGuidance        centroid + size losses → x_t["translation/6drotation_normalized/scale"]
  DepthGuidance       scale-invariant depth  → x_t["shape"]
  NormalGuidance      Sobel normal loss      → x_t["shape"]
  CompositeGuidance   chains modules; last write wins for shared keys

Usage::

    guidance = CompositeGuidance([
        ShapeGuidance(mask_path, shape_scale=5.0),
        PoseGuidance(mask_path, pose_scale=0.05),
        DepthGuidance(depth_path, depth_scale=5.0),
        NormalGuidance(depth_path, normal_scale=2.0)
    ])

    corrections = guidance.apply(
        x_t, ss_decoder, pose_decoder, intrinsics,
        scene_scale=..., scene_shift=..., t_step=float(t),
    )
    for key, val in corrections.items():
        x_t[key].data.copy_(val)

Each module returns {} when t_step <= start_t or the mesh is empty.
Gradients computed with torch.autograd.grad() — no .grad accumulation side effects.
"""

import os
from abc import ABC, abstractmethod

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
from scipy import ndimage

# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────


def _load_and_clean_mask(mask_path: str, size: int, min_blob_area: int) -> torch.Tensor:
    """Load + resize mask, drop noise blobs < min_blob_area. Returns (H,W) float32."""
    arr = np.array(Image.open(mask_path))
    mask = arr > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    mask = (
        np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (size, size), Image.NEAREST
            )
        )
        > 0
    )
    labeled, n = ndimage.label(mask)
    cleaned = np.zeros_like(mask, dtype=np.float32)
    for i in range(1, n + 1):
        if (labeled == i).sum() >= min_blob_area:
            cleaned[labeled == i] = 1.0
    return torch.from_numpy(cleaned)


def _load_gt_depth(depth_path: str, size: int = 256) -> torch.Tensor:
    """Load Open3DHOI depth.npy, resize bilinearly. Returns (H,W) float32, background=0."""
    depth = np.load(depth_path).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = torch.from_numpy(depth).float()
    return torch.nn.functional.interpolate(
        depth[None, None], size=(size, size), mode="bilinear", align_corners=False
    )[0, 0]


def _build_flexicubes_grid(N: int, device: torch.device):
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


def _extract_mesh(ss_grid: torch.Tensor, device: torch.device):
    """FlexiCubes from ss_grid (1,C,N,N,N). Returns (verts, faces) or None if empty."""
    from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.flexicubes.flexicubes import (
        FlexiCubes,
    )

    N = ss_grid.shape[-1]
    scalar_field = -ss_grid[0, 0].reshape(-1).to(device)
    if (scalar_field < 0).sum() == 0:
        return None
    grid_verts, cube_idx = _build_flexicubes_grid(N, device)
    fc = FlexiCubes(device=device)
    verts, faces, _, _ = fc(grid_verts, scalar_field, cube_idx, resolution=N - 1)
    if verts.shape[0] == 0:
        return None
    return verts, faces


def _render_soft_silhouette(
    verts,
    faces,
    pose_rotation,
    pose_translation,
    pose_scale,
    intrinsics,
    image_size: int,
    device,
):
    """Differentiable soft silhouette (H,W) in [0,1] using actual scene pose + intrinsics."""
    scale = pose_scale.float().to(device).mean()
    mesh = Meshes(
        verts=(verts.to(device) * scale).unsqueeze(0),
        faces=faces.long().to(device).unsqueeze(0),
    )
    R = quaternion_to_matrix(pose_rotation.reshape(1, 4).to(device))
    T = pose_translation.reshape(1, 3).to(device)
    K = intrinsics.to(device)
    cameras = PerspectiveCameras(
        focal_length=((K[0, 0] * image_size, K[1, 1] * image_size),),
        principal_point=((K[0, 2] * image_size, K[1, 2] * image_size),),
        R=R,
        T=T,
        in_ndc=False,
        image_size=((image_size, image_size),),
        device=device,
    )
    blend = BlendParams(sigma=1e-4, gamma=1e-4)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=image_size,
                blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend.sigma,
                faces_per_pixel=50,
            ),
        ),
        shader=SoftSilhouetteShader(blend_params=blend),
    )
    return renderer(mesh)[0, ..., 3]  # (H, W)


def _render_depth(
    verts,
    faces,
    pose_rotation,
    pose_translation,
    pose_scale,
    intrinsics,
    image_size: int,
    device,
):
    """Hard zbuf depth map (H,W). Background = -1. Uses actual scene pose + intrinsics."""
    scale = pose_scale.float().to(device).mean()
    mesh = Meshes(
        verts=(verts.to(device) * scale).unsqueeze(0),
        faces=faces.long().to(device).unsqueeze(0),
    )
    R = quaternion_to_matrix(pose_rotation.reshape(1, 4).to(device))
    T = pose_translation.reshape(1, 3).to(device)
    K = intrinsics.to(device)
    cameras = PerspectiveCameras(
        focal_length=((K[0, 0] * image_size, K[1, 1] * image_size),),
        principal_point=((K[0, 2] * image_size, K[1, 2] * image_size),),
        R=R,
        T=T,
        in_ndc=False,
        image_size=((image_size, image_size),),
        device=device,
    )
    fragments = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=image_size, blur_radius=0.0, faces_per_pixel=1
        ),
    )(mesh)
    return fragments.zbuf[0, ..., 0]  # (H, W)


def _loss_iou(pred_alpha, gt_mask):
    gt = gt_mask.to(pred_alpha.device)
    inter = (pred_alpha * gt).sum()
    union = (pred_alpha + gt - pred_alpha * gt).sum().clamp(min=1e-8)
    return 1.0 - inter / union


def _loss_centroid(pred_alpha, gt_cx, gt_cy):
    H, W = pred_alpha.shape
    device = pred_alpha.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device).float(),
        torch.arange(W, device=device).float(),
        indexing="ij",
    )
    total = pred_alpha.sum() + 1e-8
    cx = (pred_alpha * grid_x).sum() / total
    cy = (pred_alpha * grid_y).sum() / total
    return ((cx - gt_cx.to(device)) ** 2 + (cy - gt_cy.to(device)) ** 2).sqrt() / W


def _loss_size(pred_alpha, gt_area):
    pred_area = pred_alpha.sum()
    return (
        (pred_area - gt_area.to(pred_alpha.device))
        / (gt_area.to(pred_alpha.device) + 1e-8)
    ) ** 2


def _loss_depth(pred_depth, gt_depth):
    """Scale-invariant depth MSE over pixels valid in both pred and GT."""
    gt = gt_depth.to(pred_depth.device)
    valid = (gt > 0) & (pred_depth > 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device, requires_grad=True)
    pred = pred_depth[valid]
    target = gt[valid]
    pred = (pred - pred.mean()) / (pred.std() + 1e-8)
    target = (target - target.mean()) / (target.std() + 1e-8)
    return ((pred - target) ** 2).mean()


def _apply_correction(
    original: torch.Tensor, grad: torch.Tensor, scale: float, name: str
) -> torch.Tensor:
    """Unit-norm gradient step. scale is the literal step size regardless of grad magnitude."""
    if not torch.isfinite(grad).all():
        print(f"  [guidance] {name}: NaN/Inf grad — skipping")
        return original
    if grad.norm() < 1e-12:
        print(f"  [guidance] {name}: zero grad — skipping")
        return original
    g = grad / (grad.norm() + 1e-8)
    print(f"  [guidance] {name}: scale={scale:.6f}")
    return original - scale * g.to(original.device)


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────


class BaseGuidance(ABC):
    """
    Abstract base for all guidance modules.

    Contract
    --------
    apply() returns a dict containing ONLY the x_t keys this module corrects.
    Keys not in the return dict are left unchanged by the caller.
    Returns {} when t_step <= start_t or when the mesh is empty/degenerate.

    Caller pattern::

        corrections = guidance.apply(x_t, ss_decoder, pose_decoder,
                                     intrinsics, scene_scale, scene_shift, t_step)
        for key, val in corrections.items():
            x_t[key].data.copy_(val)
    """

    @abstractmethod
    def apply(
        self,
        x_t: dict,
        ss_decoder,
        pose_decoder,
        intrinsics: torch.Tensor,
        scene_scale=None,
        scene_shift=None,
        t_step: float = 1.0,
    ) -> dict: ...


# ─────────────────────────────────────────────────────────────────────────────
# ShapeGuidance
# ─────────────────────────────────────────────────────────────────────────────


class ShapeGuidance(BaseGuidance):
    """
    Soft-IoU silhouette loss → corrects x_t['shape'].

    Gradient flows: shape_lat → ss_decoder → FlexiCubes → soft render → IoU loss.
    Pose is decoded from x_t without grad (held fixed).
    """

    def __init__(
        self,
        mask_path: str,
        shape_scale: float = 5.0,
        image_size: int = 256,
        min_blob_area: int = 20,
        start_t: float = 0.0,
        device: str = "cpu",
        debug_dir: str = None,
    ):
        self.shape_scale = shape_scale
        self.image_size = image_size
        self.start_t = start_t
        self.device = torch.device(device)
        self.debug_dir = debug_dir
        self._step = 0

        self.gt_mask = _load_and_clean_mask(mask_path, image_size, min_blob_area)
        _, n_blobs = ndimage.label(self.gt_mask.numpy())
        print(
            f"[ShapeGuidance] mask={int(self.gt_mask.sum())}px  blobs={n_blobs}  "
            f"scale={shape_scale}  start_t={start_t}"
        )
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    @torch.enable_grad()
    def apply(
        self,
        x_t,
        ss_decoder,
        pose_decoder,
        intrinsics,
        scene_scale=None,
        scene_shift=None,
        t_step: float = 1.0,
    ) -> dict:
        if t_step <= self.start_t:
            return {}

        B = x_t["shape"].shape[0]
        shape_lat = x_t["shape"].detach().float().requires_grad_(True)
        ss_grid = ss_decoder(
            shape_lat.permute(0, 2, 1).contiguous().view(B, 8, 16, 16, 16)
        )

        n_occ = int((ss_grid[0, 0] > 0).sum())
        print(f"  [shape] t={t_step:.3f}  occupied={n_occ}/{ss_grid.shape[-1] ** 3}")

        result = _extract_mesh(ss_grid, self.device)
        if result is None:
            print("  [shape] empty mesh — skipping")
            return {}
        verts, faces = result

        with torch.no_grad():
            pose = pose_decoder(x_t, scene_scale=scene_scale, scene_shift=scene_shift)

        pred_alpha = _render_soft_silhouette(
            verts,
            faces,
            pose["rotation"],
            pose["translation"],
            pose["scale"],
            intrinsics,
            self.image_size,
            self.device,
        )
        loss = _loss_iou(pred_alpha, self.gt_mask)
        if not torch.isfinite(loss):
            print("  [shape] non-finite loss — skipping")
            return {}

        (grad,) = torch.autograd.grad(loss, shape_lat)
        print(
            f"  [shape] iou_loss={loss.item():.4f}  pred={pred_alpha.mean().item():.4f}"
        )
        self._save_debug(pred_alpha)
        self._step += 1

        with torch.no_grad():
            return {
                "shape": _apply_correction(
                    x_t["shape"], grad, self.shape_scale, "shape"
                )
            }

    def _save_debug(self, pred_alpha):
        if self.debug_dir is None:
            return
        pred_np = (pred_alpha.detach().cpu().numpy() * 255).astype(np.uint8)
        gt_np = (self.gt_mask.numpy() * 255).astype(np.uint8)
        pred_bin, gt_bin = pred_np > 127, gt_np > 127
        overlap = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
        overlap[pred_bin & gt_bin] = [255, 255, 255]
        overlap[pred_bin & ~gt_bin] = [255, 80, 80]
        overlap[~pred_bin & gt_bin] = [80, 80, 255]
        H, W = pred_np.shape
        canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
        canvas[:, :W] = np.stack([pred_np] * 3, axis=-1)
        canvas[:, W : 2 * W] = np.stack([gt_np] * 3, axis=-1)
        canvas[:, 2 * W :] = overlap
        Image.fromarray(canvas).save(
            os.path.join(self.debug_dir, f"shape_step_{self._step:03d}.png")
        )


# ─────────────────────────────────────────────────────────────────────────────
# PoseGuidance
# ─────────────────────────────────────────────────────────────────────────────


class PoseGuidance(BaseGuidance):
    """
    Centroid loss → corrects x_t["translation"].
    Size loss     → corrects x_t["scale"].

    Gradient flows: pose_lat → pose_decoder → camera pose → soft render → losses.
    Mesh geometry is held fixed (ss_decoder run without grad).
    6drotation is not corrected (centroid/size give no meaningful rotation signal).
    """

    def __init__(
        self,
        mask_path: str,
        pose_scale: float = 0.05,
        w_centroid: float = 1.0,
        w_size: float = 1.0,
        image_size: int = 256,
        min_blob_area: int = 20,
        start_t: float = 0.0,
        device: str = "cpu",
    ):
        self.pose_scale = pose_scale
        self.w_centroid = w_centroid
        self.w_size = w_size
        self.image_size = image_size
        self.start_t = start_t
        self.device = torch.device(device)
        self._step = 0

        self.gt_mask = _load_and_clean_mask(mask_path, image_size, min_blob_area)
        H, W = self.gt_mask.shape
        gy, gx = torch.meshgrid(
            torch.arange(H).float(), torch.arange(W).float(), indexing="ij"
        )
        total = self.gt_mask.sum() + 1e-8
        self.gt_cx = (self.gt_mask * gx).sum() / total
        self.gt_cy = (self.gt_mask * gy).sum() / total
        self.gt_area = self.gt_mask.sum()
        print(
            f"[PoseGuidance] centroid=({self.gt_cx:.1f},{self.gt_cy:.1f})  "
            f"scale={pose_scale}  start_t={start_t}"
        )

    @torch.enable_grad()
    def apply(
        self,
        x_t,
        ss_decoder,
        pose_decoder,
        intrinsics,
        scene_scale=None,
        scene_shift=None,
        t_step: float = 1.0,
    ) -> dict:
        if t_step <= self.start_t:
            return {}

        B = x_t["shape"].shape[0]

        # Fixed mesh — ss_decoder without grad
        with torch.no_grad():
            ss_grid = ss_decoder(
                x_t["shape"].permute(0, 2, 1).contiguous().view(B, 8, 16, 16, 16)
            )

        result = _extract_mesh(ss_grid, self.device)
        if result is None:
            print(f"  [pose] t={t_step:.3f}  empty mesh — skipping")
            return {}
        verts = result[0].detach()
        faces = result[1].detach()

        # Pose latents with grad — rotation excluded (no meaningful signal from centroid/size)
        trans_lat = x_t["translation"].detach().float().requires_grad_(True)
        scale_lat = x_t["scale"].detach().float().requires_grad_(True)

        x_t_pose = {**x_t, "translation": trans_lat, "scale": scale_lat}
        pose = pose_decoder(x_t_pose, scene_scale=scene_scale, scene_shift=scene_shift)

        pred_alpha = _render_soft_silhouette(
            verts,
            faces,
            pose["rotation"],
            pose["translation"],
            pose["scale"],
            intrinsics,
            self.image_size,
            self.device,
        )

        corrections = {}
        l_c = l_s = 0.0

        if self.w_centroid > 0:
            lc = _loss_centroid(pred_alpha, self.gt_cx, self.gt_cy)
            if not torch.isfinite(lc):
                print(f"  [pose] t={t_step:.3f}  non-finite centroid loss — skipping")
                return {}
            (g_trans,) = torch.autograd.grad(lc, trans_lat, retain_graph=True)
            corrections["translation"] = _apply_correction(
                x_t["translation"], g_trans, self.pose_scale, "translation"
            )
            l_c = lc.item()

        if self.w_size > 0:
            ls = _loss_size(pred_alpha, self.gt_area)
            if not torch.isfinite(ls):
                print(f"  [pose] t={t_step:.3f}  non-finite size loss — skipping")
                return {}
            (g_scale,) = torch.autograd.grad(ls, scale_lat)
            corrections["scale"] = _apply_correction(
                x_t["scale"], g_scale, self.pose_scale, "pose_scale"
            )
            l_s = ls.item()

        print(f"  [pose] t={t_step:.3f}  centroid={l_c:.4f}  size={l_s:.4f}")
        self._step += 1

        return corrections


# ─────────────────────────────────────────────────────────────────────────────
# DepthGuidance
# ─────────────────────────────────────────────────────────────────────────────


class DepthGuidance(BaseGuidance):
    """
    Scale-invariant depth loss → corrects x_t['shape'].

    GT depth from Open3DHOI depth.npy, masked to object region via mask_path.
    Gradient flows: shape_lat → ss_decoder → FlexiCubes → zbuf render → depth loss.
    Pose is decoded from x_t without grad (held fixed).
    """

    def __init__(
        self,
        depth_path: str,
        mask_path: str = None,
        depth_scale: float = 5.0,
        image_size: int = 256,
        start_t: float = 0.0,
        device: str = "cpu",
    ):
        self.depth_scale = depth_scale
        self.image_size = image_size
        self.start_t = start_t
        self.device = torch.device(device)

        gt_depth = _load_gt_depth(depth_path, image_size)
        if mask_path is not None:
            obj_mask = _load_and_clean_mask(mask_path, image_size, min_blob_area=0)
            gt_depth = gt_depth * obj_mask
        self.gt_depth = gt_depth
        print(
            f"[DepthGuidance] depth={depth_path}  mask={'yes' if mask_path else 'no'}  "
            f"scale={depth_scale}  start_t={start_t}"
        )

    @torch.enable_grad()
    def apply(
        self,
        x_t,
        ss_decoder,
        pose_decoder,
        intrinsics,
        scene_scale=None,
        scene_shift=None,
        t_step: float = 1.0,
    ) -> dict:
        if t_step <= self.start_t:
            return {}

        B = x_t["shape"].shape[0]
        shape_lat = x_t["shape"].detach().float().requires_grad_(True)
        ss_grid = ss_decoder(
            shape_lat.permute(0, 2, 1).contiguous().view(B, 8, 16, 16, 16)
        )

        n_occ = int((ss_grid[0, 0] > 0).sum())
        print(f"  [depth] t={t_step:.3f}  occupied={n_occ}/{ss_grid.shape[-1] ** 3}")

        result = _extract_mesh(ss_grid, self.device)
        if result is None:
            print(f"  [depth] t={t_step:.3f}  empty mesh — skipping")
            return {}
        verts, faces = result

        with torch.no_grad():
            pose = pose_decoder(x_t, scene_scale=scene_scale, scene_shift=scene_shift)

        pred_depth = _render_depth(
            verts,
            faces,
            pose["rotation"],
            pose["translation"],
            pose["scale"],
            intrinsics,
            self.image_size,
            self.device,
        )
        valid_px = int(
            ((self.gt_depth.to(pred_depth.device) > 0) & (pred_depth > 0)).sum()
        )
        print(f"  [depth] t={t_step:.3f}  valid_px={valid_px}")

        loss = _loss_depth(pred_depth, self.gt_depth)
        if not torch.isfinite(loss):
            print(f"  [depth] t={t_step:.3f}  non-finite loss — skipping")
            return {}

        (grad,) = torch.autograd.grad(loss, shape_lat, allow_unused=True)
        if grad is None:
            print(
                f"  [depth] t={t_step:.3f}  no overlap with masked GT depth — skipping"
            )
            return {}
        print(f"  [depth] t={t_step:.3f}  loss={loss.item():.6f}")

        with torch.no_grad():
            return {
                "shape": _apply_correction(
                    x_t["shape"], grad, self.depth_scale, "depth_shape"
                )
            }


# ─────────────────────────────────────────────────────────────────────────────
# NormalGuidance
# ─────────────────────────────────────────────────────────────────────────────


class NormalGuidance(BaseGuidance):
    """
    Sobel-normal consistency guidance → corrects x_t['shape'].

    Pipeline:
        shape_lat
            → ss_decoder
            → FlexiCubes
            → depth render
            → Sobel normals
            → cosine normal loss

    Pose is decoded from x_t without grad (held fixed).
    """

    def __init__(
        self,
        depth_path: str,
        mask_path: str = None,
        normal_scale: float = 5.0,
        image_size: int = 256,
        start_t: float = 0.0,
        device: str = "cpu",
    ):
        self.normal_scale = normal_scale
        self.image_size = image_size
        self.start_t = start_t
        self.device = torch.device(device)

        gt_depth = _load_gt_depth(depth_path, image_size)

        if mask_path is not None:
            obj_mask = _load_and_clean_mask(
                mask_path,
                image_size,
                min_blob_area=0,
            )
            gt_depth = gt_depth * obj_mask

        self.gt_depth = gt_depth

        print(
            f"[NormalGuidance] depth={depth_path}  "
            f"mask={'yes' if mask_path else 'no'}  "
            f"scale={normal_scale}  start_t={start_t}"
        )

    def _depth_to_normals(self, depth: torch.Tensor):
        """
        depth: (H,W)

        returns:
            normals: (3,H,W)
        """

        device = depth.device

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)

        d = depth[None, None]

        dzdx = torch.nn.functional.conv2d(
            d,
            sobel_x,
            padding=1,
        )[0, 0]

        dzdy = torch.nn.functional.conv2d(
            d,
            sobel_y,
            padding=1,
        )[0, 0]

        nx = -dzdx
        ny = -dzdy
        nz = torch.ones_like(depth)

        normals = torch.stack([nx, ny, nz], dim=0)

        normals = torch.nn.functional.normalize(
            normals,
            dim=0,
        )

        return normals

    def _loss_normal(self, pred_depth, gt_depth):
        """
        Surface normal consistency loss.
        """

        gt = gt_depth.to(pred_depth.device)

        valid = (gt > 0) & (pred_depth > 0)

        if valid.sum() == 0:
            return torch.tensor(
                0.0,
                device=pred_depth.device,
                requires_grad=True,
            )

        pred_n = self._depth_to_normals(pred_depth)
        gt_n = self._depth_to_normals(gt)

        cosine = (pred_n * gt_n).sum(dim=0)

        return (1.0 - cosine[valid]).mean()

    @torch.enable_grad()
    def apply(
        self,
        x_t,
        ss_decoder,
        pose_decoder,
        intrinsics,
        scene_scale=None,
        scene_shift=None,
        t_step: float = 1.0,
    ) -> dict:

        if t_step <= self.start_t:
            return {}

        B = x_t["shape"].shape[0]

        shape_lat = (
            x_t["shape"]
            .detach()
            .float()
            .requires_grad_(True)
        )

        ss_grid = ss_decoder(
            shape_lat.permute(0, 2, 1)
            .contiguous()
            .view(B, 8, 16, 16, 16)
        )

        n_occ = int((ss_grid[0, 0] > 0).sum())

        print(
            f"  [normal] t={t_step:.3f}  "
            f"occupied={n_occ}/{ss_grid.shape[-1] ** 3}"
        )

        result = _extract_mesh(ss_grid, self.device)

        if result is None:
            print(
                f"  [normal] t={t_step:.3f}  "
                f"empty mesh — skipping"
            )
            return {}

        verts, faces = result

        with torch.no_grad():
            pose = pose_decoder(
                x_t,
                scene_scale=scene_scale,
                scene_shift=scene_shift,
            )

        pred_depth = _render_depth(
            verts,
            faces,
            pose["rotation"],
            pose["translation"],
            pose["scale"],
            intrinsics,
            self.image_size,
            self.device,
        )

        valid_px = int(
            (
                (self.gt_depth.to(pred_depth.device) > 0)
                & (pred_depth > 0)
            ).sum()
        )

        print(
            f"  [normal] t={t_step:.3f}  "
            f"valid_px={valid_px}"
        )

        loss = self._loss_normal(
            pred_depth,
            self.gt_depth,
        )

        if not torch.isfinite(loss):
            print(
                f"  [normal] t={t_step:.3f}  "
                f"non-finite loss — skipping"
            )
            return {}

        (grad,) = torch.autograd.grad(
            loss,
            shape_lat,
            allow_unused=True,
        )

        if grad is None:
            print(
                f"  [normal] t={t_step:.3f}  "
                f"no overlap — skipping"
            )
            return {}

        print(
            f"  [normal] t={t_step:.3f}  "
            f"loss={loss.item():.6f}"
        )

        with torch.no_grad():
            return {
                "shape": _apply_correction(
                    x_t["shape"],
                    grad,
                    self.normal_scale,
                    "normal",
                )
            }

# ─────────────────────────────────────────────────────────────────────────────
# CompositeGuidance
# ─────────────────────────────────────────────────────────────────────────────


class CompositeGuidance(BaseGuidance):
    """
    Runs a list of BaseGuidance modules and merges their corrections.

    Each module receives the original x_t (not yet corrected by earlier modules).
    If two modules correct the same key (e.g. shape by both ShapeGuidance and
    DepthGuidance), corrections are summed: corrected = original + Δ1 + Δ2.

    Ablation examples::

        # pose only
        CompositeGuidance([PoseGuidance(...)])

        # normal only
        CompositeGuidance([NormalGuidance(...)])

        # silhouette + pose
        CompositeGuidance([ShapeGuidance(...), PoseGuidance(...)])

        # all three
        CompositeGuidance([ShapeGuidance(...), PoseGuidance(...), DepthGuidance(...)])
    """

    def __init__(self, modules: list):
        self.modules = modules

    def apply(
        self,
        x_t,
        ss_decoder,
        pose_decoder,
        intrinsics,
        scene_scale=None,
        scene_shift=None,
        t_step: float = 1.0,
    ) -> dict:
        corrections = {}
        for mod in self.modules:
            partial = mod.apply(
                x_t,
                ss_decoder,
                pose_decoder,
                intrinsics,
                scene_scale=scene_scale,
                scene_shift=scene_shift,
                t_step=t_step,
            )
            for key, val in partial.items():
                if key in corrections:
                    # Additive: sum deltas from original so both corrections apply
                    corrections[key] = corrections[key] + val - x_t[key]
                else:
                    corrections[key] = val
        return corrections
