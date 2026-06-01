"""
Render 5-panel visualization for selected instances.

Panels:
  1. Image
  2. Mask
  3. GT mesh — normalized to unit sphere, rendered from canonical front view
  4. Pred mesh — posed using pose_params.pt (MoGe scale/rotation/translation)
  5. Alpha blend — original image with pred mesh overlaid

Usage:
    python scripts/viz_worst5.py
    python scripts/viz_worst5.py --instances rerun_worst12.txt --device cuda
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from PIL import Image
from pytorch3d.renderer import (
    BlendParams, HardPhongShader,
    MeshRasterizer, MeshRenderer, PerspectiveCameras, PointLights,
    RasterizationSettings, TexturesVertex,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_ROOT = "data/Open3DHOI/data"
PRED_ROOT = "outputs/baseline_all"
OUT_DIR   = "outputs/viz_worst5"
SIZE      = 256

INSTANCES = [
    ("suitcase",    "HICO_train2015_00000809"),
    ("skis",        "COCO_train2014_000000002758"),
    ("banana",      "HICO_train2015_00010614"),
    ("chair",       "2339751"),
    ("suitcase",    "HICO_train2015_00007565"),
]

# Per-instance GT yaw (Y-axis) rotation in degrees applied after Y flip
GT_YAW = {
    ("suitcase", "HICO_train2015_00000809"):  90,
    ("skis",     "COCO_train2014_000000002758"): 180,
    ("banana",   "HICO_train2015_00010614"):  180,
    ("chair",    "2339751"):                  -90,
    ("suitcase", "HICO_train2015_00007565"):   90,
}

# Per-instance pred rotation for panel 5: (yaw_deg, pitch_deg, roll_deg)
PRED_ROT = {
    ("suitcase", "HICO_train2015_00000809"):     (180, 45, 0),
    ("skis",     "COCO_train2014_000000002758"): (  0,  0,  0),
    ("banana",   "HICO_train2015_00010614"):     (  0,  0,  0),
    ("chair",    "2339751"):                     (180, 90,  0),
    ("suitcase", "HICO_train2015_00007565"):     (180,  0,  0),
}


def make_textures(verts):
    return TexturesVertex(
        verts_features=(torch.ones_like(verts) * 0.75).unsqueeze(0)
    )


def load_mesh(path, device):
    mesh  = trimesh.load(path, force="mesh", process=False)
    verts = torch.tensor(np.array(mesh.vertices, dtype=np.float32), device=device)
    faces = torch.tensor(np.array(mesh.faces,    dtype=np.int64),   device=device)
    return verts, faces


def normalize_verts(verts):
    mn, mx = verts.min(0).values, verts.max(0).values
    center = (mn + mx) / 2
    scale  = 2.0 / (mx - mn).max()
    return (verts - center) * scale


def phong_renderer(cameras, device):
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]],
                         ambient_color=[[0.5, 0.5, 0.5]],
                         diffuse_color=[[0.5, 0.5, 0.5]],
                         specular_color=[[0.0, 0.0, 0.0]])
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=SIZE, blur_radius=0.0, faces_per_pixel=1, bin_size=0)),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights,
                               blend_params=BlendParams()),
    )


def canonical_camera(device):
    return PerspectiveCameras(
        focal_length=((300.0, 300.0),),
        principal_point=((SIZE / 2, SIZE / 2),),
        R=torch.eye(3, device=device).unsqueeze(0),
        T=torch.tensor([[0.0, 0.0, 3.0]], device=device),
        in_ndc=False,
        image_size=((SIZE, SIZE),),
        device=device,
    )


def rot_y(angle_deg, device):
    a = np.radians(angle_deg)
    return torch.tensor([
        [ np.cos(a), 0, np.sin(a)],
        [         0, 1,         0],
        [-np.sin(a), 0, np.cos(a)],
    ], dtype=torch.float32, device=device)


def render_gt_canonical(verts, faces, device, yaw_deg=0):
    """Render GT normalized mesh, Y flipped then yaw-rotated."""
    verts_n = normalize_verts(verts)
    verts_n = verts_n * torch.tensor([1, -1, 1], dtype=torch.float32, device=device)
    if yaw_deg:
        verts_n = (rot_y(yaw_deg, device) @ verts_n.T).T
    cameras = canonical_camera(device)
    renderer = phong_renderer(cameras, device)
    mesh = Meshes(verts=[verts_n], faces=[faces], textures=make_textures(verts_n))
    with torch.no_grad():
        img = renderer(mesh)
    return img[0].clamp(0, 1).cpu().numpy()


def render_pred_normalized(verts, faces, device):
    """Render pred normalized to unit sphere from canonical front view."""
    verts_n = normalize_verts(verts)
    cameras = canonical_camera(device)
    renderer = phong_renderer(cameras, device)
    mesh = Meshes(verts=[verts_n], faces=[faces], textures=make_textures(verts_n))
    with torch.no_grad():
        img = renderer(mesh)
    return img[0].clamp(0, 1).cpu().numpy()


def render_pred_posed(verts, faces, pose_params, device):
    """Render pred mesh in camera space using MoGe pose_params."""
    scale   = pose_params["scale"].to(device).float().mean()
    verts_s = verts * scale
    K       = pose_params["intrinsics"].to(device)
    R       = quaternion_to_matrix(pose_params["rotation"].reshape(1, 4).to(device))
    T       = pose_params["translation"].reshape(1, 3).to(device)
    cameras = PerspectiveCameras(
        focal_length=((K[0, 0].item() * SIZE, K[1, 1].item() * SIZE),),
        principal_point=((K[0, 2].item() * SIZE, K[1, 2].item() * SIZE),),
        R=R, T=T, in_ndc=False,
        image_size=((SIZE, SIZE),),
        device=device,
    )
    renderer = phong_renderer(cameras, device)
    mesh = Meshes(verts=[verts_s], faces=[faces], textures=make_textures(verts_s))
    with torch.no_grad():
        img = renderer(mesh)
    return img[0].clamp(0, 1).cpu().numpy()


def rot_x(angle_deg, device):
    a = np.radians(angle_deg)
    return torch.tensor([
        [1,          0,           0],
        [0,  np.cos(a), -np.sin(a)],
        [0,  np.sin(a),  np.cos(a)],
    ], dtype=torch.float32, device=device)


def rot_z(angle_deg, device):
    a = np.radians(angle_deg)
    return torch.tensor([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [        0,          0, 1],
    ], dtype=torch.float32, device=device)


def render_pred_rotated(verts, faces, device, yaw_deg=45, pitch_deg=0, roll_deg=0):
    """Render pred normalized, rotated yaw (Y) → pitch (X) → roll (Z)."""
    verts_n = normalize_verts(verts)
    verts_r = (rot_y(yaw_deg, device) @ verts_n.T).T
    if pitch_deg:
        verts_r = (rot_x(pitch_deg, device) @ verts_r.T).T
    if roll_deg:
        verts_r = (rot_z(roll_deg, device) @ verts_r.T).T
    cameras = canonical_camera(device)
    renderer = phong_renderer(cameras, device)
    mesh = Meshes(verts=[verts_r], faces=[faces], textures=make_textures(verts_r))
    with torch.no_grad():
        img = renderer(mesh)
    return img[0].clamp(0, 1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", default=None)
    parser.add_argument("--device",    default="cuda")
    args = parser.parse_args()

    instances = INSTANCES
    if args.instances:
        instances = []
        with open(args.instances) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("/", 1)
                if len(parts) == 2:
                    instances.append((parts[0], parts[1]))

    os.makedirs(OUT_DIR, exist_ok=True)
    device = args.device

    for i, (cat, inst) in enumerate(instances):
        inst_dir  = os.path.join(DATA_ROOT, cat, inst)
        gt_path   = os.path.join(inst_dir, "object_mesh.obj")
        pred_path = os.path.join(PRED_ROOT, cat, inst, "pred_mesh.obj")
        pose_path = os.path.join(PRED_ROOT, cat, inst, "pose_params.pt")
        img_path  = os.path.join(inst_dir, "image.jpg")
        mask_path = os.path.join(inst_dir, "obj_mask.png")

        print(f"[{i+1}/{len(instances)}] {cat}/{inst}", end=" ... ", flush=True)

        if not all(os.path.exists(p) for p in [gt_path, pred_path, pose_path]):
            print("SKIP — missing files")
            continue

        try:
            pose_params = torch.load(pose_path, map_location="cpu", weights_only=False)
            print(f"  pose scale={pose_params.get('scale')}, T={pose_params.get('translation')}")
            gt_v,   gt_f   = load_mesh(gt_path,   device)
            pred_v, pred_f = load_mesh(pred_path, device)

            gt_yaw         = GT_YAW.get((cat, inst), 0)
            pred_yaw, pred_pitch, pred_roll = PRED_ROT.get((cat, inst), (45, 0, 0))

            orig_img   = np.array(Image.open(img_path).convert("RGB").resize((SIZE, SIZE)))
            mask_img   = np.array(Image.open(mask_path).convert("L").resize((SIZE, SIZE)))
            gt_render  = render_gt_canonical(gt_v, gt_f, device, yaw_deg=gt_yaw)
            pred_posed = render_pred_posed(pred_v, pred_f, pose_params, device)
            pred_rot   = render_pred_rotated(pred_v, pred_f, device, yaw_deg=pred_yaw, pitch_deg=pred_pitch, roll_deg=pred_roll)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=120)
        panels = [orig_img, mask_img, gt_render[..., :3], pred_posed[..., :3], pred_rot[..., :3]]
        titles = ["Image", "Mask", f"GT (yaw={gt_yaw}°)", "Pred (posed)", f"Pred (y={pred_yaw}° x={pred_pitch}° z={pred_roll}°)"]
        for ax, img, title in zip(axes, panels, titles):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        fig.suptitle(f"{cat} / {inst}", fontsize=11)
        plt.tight_layout()

        fname = f"{i+1:02d}_{cat}_{inst}.png".replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=120, bbox_inches="tight")
        plt.close()
        print("done")

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
