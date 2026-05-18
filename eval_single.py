"""Quick eval script: Chamfer + F-score for one predicted vs GT mesh pair.

Usage:
    python eval_single.py \
        --pred outputs/<prefix>/pred_points.npy \
        --gt   data/Open3DHOI/data/<category>/<id>/object_mesh.obj
"""

import argparse
import random

import numpy as np
import torch
import trimesh
from pytorch3d.ops import iterative_closest_point, knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_gt_points(obj_path, n=20_000, seed=0, device="cuda"):
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts_t = torch.from_numpy(verts).to(device)
    faces_t = torch.from_numpy(faces).to(device)
    pt3d_mesh = Meshes(verts=[verts_t], faces=[faces_t])
    with torch.random.fork_rng(devices=[0] if device == "cuda" else []):
        torch.manual_seed(seed)
        pts = sample_points_from_meshes(pt3d_mesh, num_samples=n)[0]
    return pts


def normalize(pts):
    mn, mx = pts.min(0).values, pts.max(0).values
    center = (mn + mx) / 2
    scale = 2.0 / (mx - mn).max()
    return (pts - center) * scale


def align_icp(pred, gt, max_iterations=100):
    result = iterative_closest_point(
        pred.unsqueeze(0), gt.unsqueeze(0), max_iterations=max_iterations
    )
    return result.Xt[0]


def chamfer(a, b):
    d_ab = (
        knn_points(a.unsqueeze(0), b.unsqueeze(0), K=1)
        .dists.squeeze()
        .clamp_min(0)
        .sqrt()
    )
    d_ba = (
        knn_points(b.unsqueeze(0), a.unsqueeze(0), K=1)
        .dists.squeeze()
        .clamp_min(0)
        .sqrt()
    )
    return float((d_ab.mean() + d_ba.mean()) / 2)


def f_score(a, b, thresholds=(0.005, 0.01, 0.02, 0.05)):
    d_a = (
        knn_points(a.unsqueeze(0), b.unsqueeze(0), K=1)
        .dists.squeeze()
        .clamp_min(0)
        .sqrt()
    )
    d_b = (
        knn_points(b.unsqueeze(0), a.unsqueeze(0), K=1)
        .dists.squeeze()
        .clamp_min(0)
        .sqrt()
    )
    results = {}
    for tau in thresholds:
        p = float((d_a < tau).float().mean())
        r = float((d_b < tau).float().mean())
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        results[tau] = f1
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to pred_points.npy")
    parser.add_argument("--gt", required=True, help="Path to object_mesh.obj")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    seed_everything()

    print("Loading prediction...")
    pred = torch.from_numpy(np.load(args.pred)).float().to(device)

    print("Loading GT mesh and sampling points...")
    gt = load_gt_points(args.gt, device=device)

    print("Normalizing both to [-1, 1]...")
    pred = normalize(pred)
    gt = normalize(gt)

    if pred.shape[0] > gt.shape[0]:
        torch.manual_seed(0)
        idx = torch.randperm(pred.shape[0], device=device)[:gt.shape[0]]
        pred = pred[idx]

    print("Running ICP alignment...")
    pred = align_icp(pred, gt)

    print("Computing metrics...")
    cd = chamfer(pred, gt)
    fs = f_score(pred, gt)

    print(f"\n{'=' * 40}")
    print(f"Chamfer Distance:  {cd:.4f}")
    for tau, f1 in fs.items():
        print(f"F-score @ {tau:.3f}:   {f1:.4f}")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
