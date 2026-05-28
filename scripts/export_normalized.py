"""
Export normalized meshes for MeshLab inspection.

For each inspection folder adds:
  gt_norm.obj           — GT normalized to unit sphere
  pred_norm.obj         — pred normalized to unit sphere
  pred_norm_fixed.obj   — pred normalized + axis fix (pred Z+ up → GT Y- up)

Axis fix: 90° rotation around X: (x, y, z) → (x, -z, y)

Usage:
    python scripts/export_normalized.py
    python scripts/export_normalized.py --inspection outputs/inspection
"""

import argparse
import os

import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_single import align_icp

INSPECTION_DIR = "outputs/inspection"


def read_obj(path):
    verts, face_lines, header_lines = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith("f "):
                face_lines.append(line)
            else:
                header_lines.append(line)
    return np.array(verts, dtype=np.float32), face_lines, header_lines


def write_obj(path, verts, face_lines, header_lines):
    with open(path, "w") as f:
        for line in header_lines:
            f.write(line)
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for line in face_lines:
            f.write(line)


def normalize_verts(v):
    center = (v.max(0) + v.min(0)) / 2
    v_c    = v - center
    scale  = np.linalg.norm(v_c, axis=1).max()
    return v_c / scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspection", default=INSPECTION_DIR)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    subdirs = [os.path.join(args.inspection, s) for s in ("worst", "best")]
    subdirs = [s for s in subdirs if os.path.isdir(s)]
    if not subdirs:
        subdirs = [args.inspection]

    folders = []
    for subdir in subdirs:
        for f in sorted(os.listdir(subdir)):
            if os.path.isdir(os.path.join(subdir, f)):
                folders.append(os.path.join(subdir, f))

    print(f"Processing {len(folders)} folders...\n")

    for folder in folders:
        gt_path   = os.path.join(folder, "gt.obj")
        pred_path = os.path.join(folder, "pred.obj")

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"SKIP {folder} — missing mesh")
            continue

        gt_v,   gt_faces,   gt_hdr   = read_obj(gt_path)
        pred_v, pred_faces, pred_hdr = read_obj(pred_path)

        gt_norm   = normalize_verts(gt_v)
        pred_norm = normalize_verts(pred_v)

        # axis fix: pred Z+ (up) → GT Y- (up), 90° around X: (x,y,z) → (x,-z,y)
        pred_fixed = np.stack([pred_norm[:,0], -pred_norm[:,2], pred_norm[:,1]], axis=1)

        # ICP on top of axis fix
        pred_fixed_t = torch.tensor(pred_fixed, dtype=torch.float32, device=args.device)
        gt_norm_t    = torch.tensor(gt_norm,    dtype=torch.float32, device=args.device)
        pred_icp     = align_icp(pred_fixed_t, gt_norm_t).cpu().numpy()

        write_obj(os.path.join(folder, "gt_norm.obj"),         gt_norm,    gt_faces,   gt_hdr)
        write_obj(os.path.join(folder, "pred_norm.obj"),       pred_norm,  pred_faces, pred_hdr)
        write_obj(os.path.join(folder, "pred_norm_fixed.obj"), pred_fixed, pred_faces, pred_hdr)
        write_obj(os.path.join(folder, "pred_icp.obj"),        pred_icp,   pred_faces, pred_hdr)

        print(f"OK   {os.path.basename(folder)}")

    print(f"\nDone. In each folder:")
    print(f"  gt_norm.obj          — GT (unit sphere)")
    print(f"  pred_norm.obj        — pred (unit sphere, original orientation)")
    print(f"  pred_norm_fixed.obj  — pred (unit sphere, axis corrected)")
    print(f"  pred_icp.obj         — pred (axis corrected + ICP aligned to GT)")


if __name__ == "__main__":
    main()
