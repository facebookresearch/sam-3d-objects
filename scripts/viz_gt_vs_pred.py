"""
Render GT vs Pred mesh side by side for the top-N worst CD instances.
5 panels: Image | Mask | GT (camera space) | Pred (MoGe posed) | Pred silhouette

GT coordinate system (Open3DHOI):
  - extrinsic = identity  →  world frame = camera frame
  - object_mesh.obj stores vertices in person-centric local space
  - cam_trans from smplx_parameters.json translates to absolute camera space
  - intrinsics from calibration.json (K[0,0] ≈ 11425 px for 800×512 image)

Pred coordinate system (SAM3D / MoGe):
  - pred_mesh.obj is in SAM3D canonical space (~unit sphere)
  - pose_params.pt: scale → metric, rotation + translation → camera space

Usage:
    python scripts/viz_gt_vs_pred.py
    python scripts/viz_gt_vs_pred.py --n 50 --device cuda
    python scripts/viz_gt_vs_pred.py --single "air cushion" floating_94
"""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from PIL import Image
from pytorch3d.renderer import (
    BlendParams, HardPhongShader,
    MeshRasterizer, MeshRenderer, PerspectiveCameras, PointLights,
    RasterizationSettings, SoftSilhouetteShader, TexturesVertex,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix

DATA_ROOT  = "data/Open3DHOI/data"
PRED_ROOT  = "outputs/baseline_all"
OUT_DIR    = "outputs/viz_gt_vs_pred"

IMAGE_SIZE      = 256   # render size for all panels
ORIG_IMG_WIDTH  = 800   # Open3DHOI image width (used to scale intrinsics)


def make_textures(verts):
    return TexturesVertex(
        verts_features=(torch.ones_like(verts) * 0.75).unsqueeze(0)
    )


# ── Mesh loaders ──────────────────────────────────────────────────────────────

def load_mesh_verts(obj_path, device):
    """Load mesh → (verts, faces) tensors, no normalization."""
    mesh  = trimesh.load(obj_path, force="mesh", process=False)
    verts = torch.tensor(np.array(mesh.vertices, dtype=np.float32), device=device)
    faces = torch.tensor(np.array(mesh.faces,    dtype=np.int64),   device=device)
    return verts, faces


# ── Camera builders ───────────────────────────────────────────────────────────

def _make_cameras_actual(inst_dir, device):
    """Build PerspectiveCameras from calibration.json (R=I, T=0).
    Intrinsics are scaled from ORIG_IMG_WIDTH to IMAGE_SIZE.
    Returns (cameras, cam_trans_tensor).
    """
    with open(os.path.join(inst_dir, "calibration.json")) as f:
        calib = json.load(f)
    with open(os.path.join(inst_dir, "smplx_parameters.json")) as f:
        smplx = json.load(f)

    scale = IMAGE_SIZE / ORIG_IMG_WIDTH
    fx    = calib["K"][0][0] * scale
    cx    = calib["K"][0][2] * scale
    cy    = calib["K"][1][2] * scale

    cam_trans = torch.tensor(smplx["cam_trans"], dtype=torch.float32, device=device)

    cameras = PerspectiveCameras(
        focal_length=((fx, fx),),
        principal_point=((cx, cy),),
        R=torch.eye(3, device=device).unsqueeze(0),
        T=torch.zeros(1, 3, device=device),
        in_ndc=False,
        image_size=((IMAGE_SIZE, IMAGE_SIZE),),
        device=device,
    )
    return cameras, cam_trans


def _make_cameras_moge(pose_params, device):
    """Build PerspectiveCameras from MoGe pose_params (for pred rendering)."""
    K = pose_params["intrinsics"].to(device)
    R = quaternion_to_matrix(
            pose_params["rotation"].reshape(1, 4).to(device))
    T = pose_params["translation"].reshape(1, 3).to(device)
    return PerspectiveCameras(
        focal_length=((K[0, 0].item() * IMAGE_SIZE, K[1, 1].item() * IMAGE_SIZE),),
        principal_point=((K[0, 2].item() * IMAGE_SIZE, K[1, 2].item() * IMAGE_SIZE),),
        R=R, T=T,
        in_ndc=False,
        image_size=((IMAGE_SIZE, IMAGE_SIZE),),
        device=device,
    )


# ── Renderers ────────────────────────────────────────────────────────────────

def _phong_renderer(cameras, device):
    lights = PointLights(device=device, location=[[0.0, 0.0, -1.0]],
                         ambient_color=[[0.6, 0.6, 0.6]],
                         diffuse_color=[[0.4, 0.4, 0.4]],
                         specular_color=[[0.0, 0.0, 0.0]])
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=IMAGE_SIZE, blur_radius=0.0, faces_per_pixel=1, bin_size=0)),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights,
                               blend_params=BlendParams()),
    )


def render_gt_camera_space(gt_verts_local, gt_faces, inst_dir, device):
    """Render GT mesh in camera space.
    Adds cam_trans (from smplx_parameters.json) to put verts in absolute camera frame,
    then renders with actual calibration intrinsics scaled to IMAGE_SIZE.
    """
    cameras, cam_trans = _make_cameras_actual(inst_dir, device)
    gt_verts_cam = gt_verts_local + cam_trans   # person-relative → camera frame
    renderer = _phong_renderer(cameras, device)
    mesh = Meshes(verts=[gt_verts_cam], faces=[gt_faces],
                  textures=make_textures(gt_verts_cam))
    with torch.no_grad():
        img = renderer(mesh)
    return img[0].clamp(0, 1).cpu().numpy()


def render_pred_phong(pred_verts_raw, pred_faces, pose_params, device):
    """Phong render of pred using MoGe pose (scale * verts → camera space)."""
    scale    = pose_params["scale"].to(device).float().mean()
    verts    = pred_verts_raw * scale
    cameras  = _make_cameras_moge(pose_params, device)
    renderer = _phong_renderer(cameras, device)
    mesh = Meshes(verts=[verts], faces=[pred_faces], textures=make_textures(verts))
    with torch.no_grad():
        img = renderer(mesh)
    return img[0].clamp(0, 1).cpu().numpy()


def render_pred_silhouette(pred_verts_raw, pred_faces, pose_params, device):
    """Soft silhouette of pred — same as guidance.py ShapeGuidance."""
    scale   = pose_params["scale"].to(device).float().mean()
    verts   = pred_verts_raw * scale
    cameras = _make_cameras_moge(pose_params, device)
    blend   = BlendParams(sigma=1e-4, gamma=1e-4)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=IMAGE_SIZE,
                blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend.sigma,
                faces_per_pixel=50,
                bin_size=0,
            )),
        shader=SoftSilhouetteShader(blend_params=blend),
    )
    mesh = Meshes(verts=[verts], faces=[pred_faces])
    with torch.no_grad():
        sil = renderer(mesh)[0, ..., 3]
    return sil.cpu().numpy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default=os.path.join(PRED_ROOT, "results_multi_init.csv"))
    parser.add_argument("--n",         type=int, default=20)
    parser.add_argument("--single",    nargs=2, metavar=("CAT", "INST"))
    parser.add_argument("--instances", default=None,
                        help="text file with cat/instance per line (same format as rerun_worst12.txt)")
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = args.device

    if args.single:
        cat, inst = args.single
        rows = [{"category": cat, "instance": inst, "chamfer": 0.0}]
    elif args.instances:
        rows = []
        csv_cds = {}
        if os.path.exists(args.csv):
            with open(args.csv) as f:
                for r in csv.DictReader(f):
                    csv_cds[(r["category"], r["instance"])] = float(r["chamfer"])
        with open(args.instances) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("/", 1)
                if len(parts) == 2:
                    cat, inst = parts
                    rows.append({"category": cat, "instance": inst,
                                 "chamfer": csv_cds.get((cat, inst), 0.0)})
    else:
        with open(args.csv) as f:
            rows = list(csv.DictReader(f))
        rows = rows[:args.n]

    print(f"Visualizing {len(rows)} instances\n")

    for i, row in enumerate(rows):
        cat  = row["category"]
        inst = row["instance"]
        cd   = float(row["chamfer"])

        inst_dir  = os.path.join(DATA_ROOT, cat, inst)
        gt_path   = os.path.join(inst_dir, "object_mesh.obj")
        pred_path = os.path.join(PRED_ROOT, cat, inst, "pred_mesh.obj")
        pose_path = os.path.join(PRED_ROOT, cat, inst, "pose_params.pt")
        img_path  = os.path.join(inst_dir, "image.jpg")
        mask_path = os.path.join(inst_dir, "obj_mask.png")

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"[{i+1}] SKIP {cat}/{inst} — missing mesh")
            continue
        if not os.path.exists(pose_path):
            print(f"[{i+1}] SKIP {cat}/{inst} — missing pose_params.pt")
            continue

        print(f"[{i+1}/{len(rows)}] {cat}/{inst}  CD={cd:.4f}", end=" ... ", flush=True)

        try:
            pose_params = torch.load(pose_path, map_location="cpu",
                                     weights_only=False)
            gt_verts,   gt_faces   = load_mesh_verts(gt_path,   device)
            pred_verts, pred_faces = load_mesh_verts(pred_path, device)

            gt_img   = render_gt_camera_space(gt_verts, gt_faces, inst_dir, device)
            pred_img = render_pred_phong(pred_verts, pred_faces, pose_params, device)
            sil_img  = render_pred_silhouette(pred_verts, pred_faces, pose_params, device)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        panels, titles = [], []
        if os.path.exists(img_path):
            panels.append(np.array(Image.open(img_path).convert("RGB"))); titles.append("Image")
        if os.path.exists(mask_path):
            panels.append(np.array(Image.open(mask_path).convert("L")));  titles.append("Mask (GT)")
        panels += [gt_img, pred_img, sil_img]
        titles += ["GT (camera space)", "Pred (MoGe posed)", "Pred silhouette"]

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), dpi=120)
        for ax, img, title in zip(axes, panels, titles):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title, fontsize=13)
            ax.axis("off")
        fig.suptitle(f"{cat} / {inst}   CD={cd:.4f}", fontsize=11)
        plt.tight_layout()

        fname = f"{i+1:03d}_{cat}_{inst}.png".replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=120, bbox_inches="tight")
        plt.close()
        print("done")

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
