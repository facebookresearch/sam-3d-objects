"""
Export normalized meshes for MeshLab inspection.

For each inspection folder adds:
  gt_norm.obj      — GT normalized to unit sphere
  pred_norm.obj    — pred normalized to unit sphere
  pred_icp.obj     — pred normalized + best axis rotation + ICP aligned to GT
  best_rotation.txt — index and matrix of the best axis-aligned rotation found

Alignment: tries all 24 axis-aligned rotations (cube symmetry group), runs ICP
from each, keeps the one with the lowest Chamfer Distance.

Usage:
    python scripts/export_normalized.py
    python scripts/export_normalized.py --inspection outputs/inspection
    python scripts/export_normalized.py --device cuda
"""

import argparse
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_single import align_icp, chamfer, normalize as normalize_pts

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


def normalize_verts(v, device="cpu"):
    """Same convention as eval_single.normalize: fits into [-1, 1] bounding box."""
    t = torch.tensor(v, dtype=torch.float32, device=device)
    return normalize_pts(t).cpu().numpy()


def axis_aligned_rotations(device):
    """All 24 proper axis-aligned rotation matrices (cube symmetry group)."""
    mats = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3), dtype=np.float32)
            for row, (col, s) in enumerate(zip(perm, signs)):
                R[row, col] = float(s)
            if np.linalg.det(R) > 0.5:
                mats.append((R, torch.tensor(R, device=device)))
    return mats


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

    rotations = axis_aligned_rotations(args.device)

    for folder in folders:
        gt_path   = os.path.join(folder, "gt.obj")
        pred_path = os.path.join(folder, "pred.obj")

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"SKIP {folder} — missing mesh")
            continue

        gt_v,   gt_faces,   gt_hdr   = read_obj(gt_path)
        pred_v, pred_faces, pred_hdr = read_obj(pred_path)

        gt_norm   = normalize_verts(gt_v,   device=args.device)
        pred_norm = normalize_verts(pred_v, device=args.device)

        gt_t   = torch.tensor(gt_norm,   dtype=torch.float32, device=args.device)
        pred_t = torch.tensor(pred_norm, dtype=torch.float32, device=args.device)

        # try all 24 axis-aligned rotations, keep best ICP result
        best_cd   = float("inf")
        best_icp  = None
        best_R_np = None
        best_idx  = 0
        for idx, (R_np, R_t) in enumerate(rotations):
            pred_rot = (R_t @ pred_t.T).T
            pred_icp = align_icp(pred_rot, gt_t)
            cd_val   = float(chamfer(pred_icp, gt_t))
            if cd_val < best_cd:
                best_cd   = cd_val
                best_icp  = pred_icp.cpu().numpy()
                best_R_np = R_np
                best_idx  = idx
            if best_cd < 0.1:
                break

        write_obj(os.path.join(folder, "gt_norm.obj"),   gt_norm,  gt_faces,   gt_hdr)
        write_obj(os.path.join(folder, "pred_norm.obj"), pred_norm, pred_faces, pred_hdr)
        write_obj(os.path.join(folder, "pred_icp.obj"),  best_icp,  pred_faces, pred_hdr)

        with open(os.path.join(folder, "best_rotation.txt"), "w") as f:
            f.write(f"rotation_index: {best_idx}\n")
            f.write(f"cd: {best_cd:.6f}\n")
            f.write(f"matrix:\n{best_R_np}\n")

        print(f"OK   {os.path.basename(folder)}  CD={best_cd:.4f}  rot={best_idx}")

    print(f"\nDone. In each folder:")
    print(f"  gt_norm.obj        — GT (unit sphere)")
    print(f"  pred_norm.obj      — pred (unit sphere, original orientation)")
    print(f"  pred_icp.obj       — pred (best axis rotation + ICP aligned to GT)")
    print(f"  best_rotation.txt  — rotation index, CD, and matrix")


if __name__ == "__main__":
    main()
