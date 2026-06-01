"""
Evaluate all baseline_all predictions vs GT meshes.
Computes Chamfer distance + F-score, saves results sorted by CD descending
(worst first) so you can immediately see where SAM3D fails.

Usage:
    python eval_baseline_all.py
    python eval_baseline_all.py --device cpu
    python eval_baseline_all.py --out outputs/baseline_all/results.csv
"""

import argparse
import csv
import itertools
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.append("notebook")
from eval_single import align_icp, chamfer, load_gt_points, normalize, seed_everything


def axis_aligned_rotations(device):
    """All 24 proper axis-aligned rotation matrices (cube symmetry group)."""
    mats = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3), dtype=np.float32)
            for row, (col, s) in enumerate(zip(perm, signs)):
                R[row, col] = float(s)
            if np.linalg.det(R) > 0.5:
                mats.append(torch.tensor(R, device=device))
    return mats

DATA_ROOT = "data/Open3DHOI/data"
PRED_ROOT = "outputs/baseline_all"


def find_pairs(pred_root):
    """Yield (category, instance, pred_mesh_path, gt_mesh_path)."""
    for cat in sorted(os.listdir(pred_root)):
        cat_dir = os.path.join(pred_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for inst in sorted(os.listdir(cat_dir)):
            pred_mesh = os.path.join(cat_dir, inst, "pred_mesh.obj")
            gt_mesh   = os.path.join(DATA_ROOT, cat, inst, "object_mesh.obj")
            if os.path.exists(pred_mesh) and os.path.exists(gt_mesh):
                yield cat, inst, pred_mesh, gt_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--pred-root", default=PRED_ROOT,
                        help="root dir containing <cat>/<inst>/pred_mesh.obj")
    parser.add_argument("--out",       default=None,
                        help="output CSV path (default: <pred-root>/results_multi_init.csv)")
    args = parser.parse_args()

    pred_root = args.pred_root
    out_csv   = args.out or os.path.join(pred_root, "results_multi_init.csv")

    seed_everything()

    pairs = list(find_pairs(pred_root))
    print(f"Found {len(pairs)} prediction/GT pairs.\n")

    rotations = axis_aligned_rotations(args.device)

    rows = []
    for i, (cat, inst, pred_path, gt_path) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {cat}/{inst} ...", end=" ", flush=True)
        try:
            pred = load_gt_points(pred_path, device=args.device)
            gt   = load_gt_points(gt_path,   device=args.device)

            pred = normalize(pred)
            gt   = normalize(gt)
            # try all 24 axis-aligned rotations, keep best ICP result
            best_cd = float("inf")
            for R in rotations:
                pred_rot = (R @ pred.T).T
                pred_icp = align_icp(pred_rot, gt)
                cd_val   = float(chamfer(pred_icp, gt))
                if cd_val < best_cd:
                    best_cd = cd_val
                if best_cd < 0.1:
                    break
            cd = best_cd
            print(f"CD={cd:.4f}")
            row = {"category": cat, "instance": inst, "chamfer": cd}
            rows.append(row)
        except Exception as e:
            print(f"ERROR: {e}")

    # Sort worst first
    rows.sort(key=lambda r: r["chamfer"], reverse=True)

    # Save CSV
    fieldnames = ["category", "instance", "chamfer"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} results to {out_csv}")

    # Top 30 worst
    print(f"\n{'='*70}")
    print(f"TOP 30 WORST (highest Chamfer Distance):")
    print(f"{'='*70}")
    print(f"{'category':<22} {'instance':<32} {'CD':>7}")
    print(f"{'-'*65}")
    for r in rows[:30]:
        print(f"{r['category']:<22} {r['instance']:<32} {r['chamfer']:>7.4f}")

    # Per-category mean CD (worst first)
    cat_cds = defaultdict(list)
    for r in rows:
        cat_cds[r["category"]].append(r["chamfer"])
    cat_means = sorted(cat_cds.items(), key=lambda x: np.mean(x[1]), reverse=True)

    print(f"\n{'='*50}")
    print(f"PER-CATEGORY MEAN CD (worst first, top 20):")
    print(f"{'='*50}")
    print(f"{'category':<25} {'mean CD':>8}  {'n':>4}")
    print(f"{'-'*50}")
    for cat, cds in cat_means[:20]:
        print(f"{cat:<25} {np.mean(cds):>8.4f}  {len(cds):>4}")

    # Overall stats
    all_cds = [r["chamfer"] for r in rows]
    print(f"\nOverall: mean CD={np.mean(all_cds):.4f}  median={np.median(all_cds):.4f}  "
          f"p90={np.percentile(all_cds, 90):.4f}  n={len(all_cds)}")


if __name__ == "__main__":
    main()
