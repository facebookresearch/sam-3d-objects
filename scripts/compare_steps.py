"""
Compare intermediate diffusion steps against GT mask and depth.

Layout (2×4 panels per step):
  Row 1 — Shape/Pose:
    original | GT mask + centroid + bbox | predicted sil + centroid + bbox | overlap
    Metrics: IoU, centroid distance (px), bbox area ratio
  Row 2 — Depth (skipped if --depth-map not provided):
    original | GT depth | predicted depth | absolute error

Usage:
    uv run python scripts/compare_steps.py \
        --steps-dir outputs/wrench_pose_005 \
        --mask data/Open3DHOI/data/wrench/wrench/obj_mask.png \
        --image data/Open3DHOI/data/wrench/wrench/image.jpg \
        --depth-map data/Open3DHOI/data/wrench/wrench/depth.npy
"""

import glob
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.flexicubes.flexicubes import (
    FlexiCubes,
)


# ── Grid + mesh ───────────────────────────────────────────────────────────────


def _build_flexicubes_grid(N, device):
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
    cube_idx = torch.stack([
        flat_idx[i, j, k],     flat_idx[i+1, j, k],
        flat_idx[i, j+1, k],   flat_idx[i+1, j+1, k],
        flat_idx[i, j, k+1],   flat_idx[i+1, j, k+1],
        flat_idx[i, j+1, k+1], flat_idx[i+1, j+1, k+1],
    ], dim=-1)
    return voxelgrid_vertices, cube_idx


def extract_mesh(ss_grid, device="cpu"):
    dev = torch.device(device)
    N = ss_grid.shape[-1]
    scalar_field = -ss_grid[0, 0].float().to(dev).reshape(-1)
    if (scalar_field < 0).sum() == 0:
        return None
    verts_grid, cube_idx = _build_flexicubes_grid(N, dev)
    fc = FlexiCubes(device=dev)
    with torch.no_grad():
        verts, faces, _, _ = fc(verts_grid, scalar_field, cube_idx, resolution=N - 1)
    return (verts, faces) if verts.shape[0] > 0 else None


# ── Rendering ─────────────────────────────────────────────────────────────────


def _make_cameras(step_data, image_size, device):
    dev = torch.device(device)
    K = step_data["intrinsics"].to(dev)
    quat = step_data["pose_rotation"].reshape(1, 4).to(dev)
    R = quaternion_to_matrix(quat)
    T = step_data["pose_translation"].reshape(1, 3).to(dev)
    return PerspectiveCameras(
        focal_length=((K[0, 0] * image_size, K[1, 1] * image_size),),
        principal_point=((K[0, 2] * image_size, K[1, 2] * image_size),),
        R=R, T=T, in_ndc=False,
        image_size=((image_size, image_size),),
        device=dev,
    )


def _scaled_mesh(verts, faces, step_data, device):
    dev = torch.device(device)
    scale = step_data["pose_scale"].float().to(dev).mean() if "pose_scale" in step_data else 1.0
    return Meshes(
        verts=(verts.to(dev) * scale).unsqueeze(0),
        faces=faces.to(dev).long().unsqueeze(0),
    )


def render_silhouette(verts, faces, step_data, image_size=512, device="cpu"):
    dev = torch.device(device)
    mesh = _scaled_mesh(verts, faces, step_data, device)
    cameras = _make_cameras(step_data, image_size, device)
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
    with torch.no_grad():
        out = renderer(mesh)
    return (out[0, ..., 3].cpu().numpy() > 0.5)


def render_depth(verts, faces, step_data, image_size=512, device="cpu"):
    """Returns (H, W) float array. Background = -1."""
    dev = torch.device(device)
    mesh = _scaled_mesh(verts, faces, step_data, device)
    cameras = _make_cameras(step_data, image_size, device)
    fragments = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1),
    )(mesh)
    return fragments.zbuf[0, ..., 0].detach().cpu().numpy()


# ── Mask + metric helpers ─────────────────────────────────────────────────────


def load_gt_mask(mask_path, size=512):
    arr = np.array(Image.open(mask_path))
    mask = arr > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 0


def load_gt_depth(depth_path, size=512):
    depth = np.load(depth_path).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return np.array(Image.fromarray(depth).resize((size, size), Image.BILINEAR))


def compute_iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    return float((pred & gt).sum()) / float((pred | gt).sum() + 1e-8)


def _centroid_bbox(mask):
    """Returns (cx, cy), (x0, y0, x1, y1) from a bool mask. None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    cx, cy = xs.mean(), ys.mean()
    return (cx, cy), (xs.min(), ys.min(), xs.max(), ys.max())


def _draw_centroid_bbox(ax, mask, color):
    centroid, bbox = _centroid_bbox(mask)
    if centroid is None:
        return None, None
    cx, cy = centroid
    x0, y0, x1, y1 = bbox
    ax.plot(cx, cy, "+", color=color, markersize=12, markeredgewidth=2)
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    return centroid, bbox


# ── Visualisation ─────────────────────────────────────────────────────────────


def save_comparison(
    silhouette, gt_mask, t_step, step_data, output_path,
    image=None, gt_depth=None, pred_depth_raw=None, image_size=512,
):
    has_depth = gt_depth is not None and pred_depth_raw is not None
    n_rows = 2 if has_depth else 1
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # ── Row 1: shape / pose ────────────────────────────────────────────────
    # Panel 0: original
    ax = axes[0, 0]
    if image is not None:
        ax.imshow(image)
        ax.set_title("Input image")
    else:
        ax.imshow(gt_mask, cmap="gray")
        ax.set_title("GT mask")
    ax.axis("off")

    # Panel 1: GT mask + centroid + bbox
    ax = axes[0, 1]
    ax.imshow(gt_mask, cmap="gray")
    gt_centroid, gt_bbox = _draw_centroid_bbox(ax, gt_mask, color="cyan")
    ax.set_title("GT mask")
    ax.axis("off")

    # Panel 2: predicted silhouette + centroid + bbox
    ax = axes[0, 2]
    ax.imshow(silhouette, cmap="gray")
    pred_centroid, pred_bbox = _draw_centroid_bbox(ax, silhouette, color="orange")
    ax.set_title("Predicted silhouette")
    ax.axis("off")

    # Panel 3: overlap
    ax = axes[0, 3]
    overlap = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    overlap[silhouette & gt_mask]  = [255, 255, 255]   # TP
    overlap[silhouette & ~gt_mask] = [255,  80,  80]   # FP
    overlap[~silhouette & gt_mask] = [ 80,  80, 255]   # FN
    ax.imshow(overlap)

    # Metrics
    iou = compute_iou(silhouette, gt_mask)
    metrics = [f"IoU={iou:.3f}"]
    if gt_centroid and pred_centroid:
        dist = np.sqrt((gt_centroid[0] - pred_centroid[0])**2 + (gt_centroid[1] - pred_centroid[1])**2)
        metrics.append(f"centroid_dist={dist:.1f}px")
    if gt_bbox and pred_bbox:
        gt_area  = (gt_bbox[2]   - gt_bbox[0])   * (gt_bbox[3]   - gt_bbox[1])
        pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
        area_ratio = pred_area / (gt_area + 1e-8)
        metrics.append(f"bbox_area_ratio={area_ratio:.2f}")
    ax.set_title("Overlap  (W=TP  R=FP  B=FN)")
    ax.axis("off")

    fig.suptitle(f"t={t_step:.3f}  " + "  ".join(metrics), fontsize=11)

    # ── Row 2: depth ──────────────────────────────────────────────────────
    if has_depth:
        # Panel 0: original again
        ax = axes[1, 0]
        if image is not None:
            ax.imshow(image)
        else:
            ax.axis("off")
        ax.set_title("Input image")
        ax.axis("off")

        # Panel 1: GT depth
        ax = axes[1, 1]
        gt_valid = gt_depth.copy()
        gt_valid[gt_valid <= 0] = np.nan
        im = ax.imshow(gt_valid, cmap="viridis")
        ax.set_title("GT depth (masked)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Panel 2: predicted depth
        ax = axes[1, 2]
        pred_vis = pred_depth_raw.copy()
        pred_vis[pred_vis < 0] = np.nan
        im2 = ax.imshow(pred_vis, cmap="viridis")
        ax.set_title("Predicted depth")
        ax.axis("off")
        plt.colorbar(im2, ax=ax, fraction=0.046)

        # Panel 3: absolute error (scale-invariant)
        ax = axes[1, 3]
        gt_valid_mask  = gt_depth > 0
        pred_valid_mask = pred_depth_raw > 0
        valid = gt_valid_mask & pred_valid_mask
        err_map = np.zeros_like(gt_depth)
        if valid.sum() > 0:
            p = pred_depth_raw[valid]
            g = gt_depth[valid]
            p_norm = (p - p.mean()) / (p.std() + 1e-8)
            g_norm = (g - g.mean()) / (g.std() + 1e-8)
            err_map[valid] = np.abs(p_norm - g_norm)
        im3 = ax.imshow(err_map, cmap="hot")
        ax.set_title("Depth error (scale-inv)")
        ax.axis("off")
        plt.colorbar(im3, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return iou


def save_gif(png_paths, output_path, duration_ms=500):
    frames = [Image.open(p).convert("RGBA") for p in png_paths]
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    print(f"GIF saved to: {output_path}  ({len(frames)} frames)")


# ── Entry point ───────────────────────────────────────────────────────────────


def main(steps_dir, mask_path, image_path=None, depth_path=None,
         image_size=512, device="cpu", gif_duration_ms=500):
    os.makedirs(steps_dir, exist_ok=True)
    pt_files = sorted(glob.glob(os.path.join(steps_dir, "step_*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No step_*.pt files found in {steps_dir}")

    gt_mask = load_gt_mask(mask_path, size=image_size)
    gt_depth = None
    if depth_path:
        gt_depth = load_gt_depth(depth_path, size=image_size)
        gt_depth = gt_depth * gt_mask  # zero out non-object pixels

    image = None
    if image_path:
        image = np.array(Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR))

    saved_pngs = []
    for pt_path in pt_files:
        step_data = torch.load(pt_path, map_location="cpu", weights_only=False)
        t_step = float(step_data["t_step"])
        print(f"t={t_step:.3f}", end="  ")

        mesh = extract_mesh(step_data["ss_grid"], device=device)
        if mesh is None:
            print("no mesh")
            continue
        verts, faces = mesh

        silhouette = render_silhouette(verts, faces, step_data, image_size, device)

        pred_depth_raw = None
        if gt_depth is not None:
            pred_depth_raw = render_depth(verts, faces, step_data, image_size, device)

        out_path = os.path.join(steps_dir, f"step_{t_step:.3f}.png")
        iou = save_comparison(
            silhouette, gt_mask, t_step, step_data, out_path,
            image=image, gt_depth=gt_depth, pred_depth_raw=pred_depth_raw,
            image_size=image_size,
        )
        print(f"IoU={iou:.3f}")
        saved_pngs.append(out_path)

    if saved_pngs:
        save_gif(saved_pngs, os.path.join(steps_dir, "steps.gif"), duration_ms=gif_duration_ms)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-dir", default="outputs/ss_steps")
    parser.add_argument("--mask", dest="mask_path", required=True)
    parser.add_argument("--image", dest="image_path", default=None)
    parser.add_argument("--depth-map", dest="depth_path", default=None,
                        help="GT depth .npy (Open3DHOI format). Enables depth row.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gif-duration", type=int, default=500, dest="gif_duration_ms")
    args = parser.parse_args()

    main(
        steps_dir=args.steps_dir,
        mask_path=args.mask_path,
        image_path=args.image_path,
        depth_path=args.depth_path,
        image_size=args.image_size,
        device=args.device,
        gif_duration_ms=args.gif_duration_ms,
    )
