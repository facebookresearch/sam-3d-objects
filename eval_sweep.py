"""
Aggregate eval over all sweep outputs.

Reads outputs/sweep/<config>/<sample>/pred_points.npy, computes
Chamfer + F-score against the GT mesh, and prints a summary table.
Results are also saved to outputs/sweep/results.csv.

Usage:
    python eval_sweep.py
    python eval_sweep.py --device cpu
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.append("notebook")  # for inference imports used transitively
from eval_single import chamfer, f_score, load_gt_points, normalize, align_icp

def load_pred_points(pred_dir, device, n=100_000):
    mesh_path = os.path.join(pred_dir, "pred_mesh.obj")
    npy_path  = os.path.join(pred_dir, "pred_points.npy")
    if os.path.exists(mesh_path):
        return load_gt_points(mesh_path, n=n, device=device), "mesh"
    elif os.path.exists(npy_path):
        return torch.from_numpy(np.load(npy_path)).float().to(device), "npy"
    return None, None

# Must match sweep.py exactly
DATA = "data/Open3DHOI/data"

SAMPLES = [
    {"name": "coffee_cup",  "dir": f"{DATA}/coffee cup/drinking_98"},
    {"name": "mug",         "dir": f"{DATA}/mug/tasting_77"},
    {"name": "bicycle",     "dir": f"{DATA}/bicycle/2413206"},
    {"name": "wine_glass",  "dir": f"{DATA}/wine glass/0c72ab77349e218f"},
    {"name": "wrench",      "dir": f"{DATA}/wrench/wrench"},
    {"name": "bucket",      "dir": f"{DATA}/bucket/bucket"},
    {"name": "helmet",      "dir": f"{DATA}/helmet/helmet"},
    {"name": "pot",         "dir": f"{DATA}/pot/pot"},
    {"name": "cup",         "dir": f"{DATA}/cup/7d9b247c9f824f5e"},
    {"name": "chair",       "dir": f"{DATA}/chair/2398495"},
]

FSCORE_THRESHOLDS = (0.005, 0.01, 0.02, 0.05)


def eval_one(pred_dir, gt_obj_path, device):
    pred, src = load_pred_points(pred_dir, device)
    gt        = load_gt_points(gt_obj_path, device=device)
    pred      = normalize(pred)
    gt        = normalize(gt)
    if pred.shape[0] > gt.shape[0]:
        idx  = torch.randperm(pred.shape[0], device=device)[:gt.shape[0]]
        pred = pred[idx]
    pred = align_icp(pred, gt)
    cd   = chamfer(pred, gt)
    fs   = f_score(pred, gt, thresholds=FSCORE_THRESHOLDS)
    return cd, fs, src


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default="outputs/sweep")
    parser.add_argument("--device",    default="cuda")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    device    = args.device

    # Discover configs from directory
    configs = sorted([
        d for d in os.listdir(sweep_dir)
        if os.path.isdir(os.path.join(sweep_dir, d)) and d != "results.csv"
    ])

    if not configs:
        print(f"No configs found in {sweep_dir}")
        sys.exit(1)

    print(f"Found {len(configs)} configs: {configs}\n")

    rows = []  # list of dicts for CSV

    for cfg_name in configs:
        cfg_results = []
        for sample in SAMPLES:
            pred_dir = os.path.join(sweep_dir, cfg_name, sample["name"])
            gt_path  = os.path.join(sample["dir"], "object_mesh.obj")

            if not os.path.exists(os.path.join(pred_dir, "pred_mesh.obj")) and \
               not os.path.exists(os.path.join(pred_dir, "pred_points.npy")):
                print(f"  SKIP {cfg_name}/{sample['name']} — no output found")
                continue

            print(f"  {cfg_name} / {sample['name']} ...", end=" ", flush=True)
            cd, fs, src = eval_one(pred_dir, gt_path, device)
            print(f"CD={cd:.4f}  F@0.02={fs[0.02]:.4f}  [{src}]")

            row = {"config": cfg_name, "sample": sample["name"], "chamfer": cd}
            row.update({f"f@{tau}": fs[tau] for tau in FSCORE_THRESHOLDS})
            rows.append(row)
            cfg_results.append((cd, fs))

        if cfg_results:
            mean_cd = np.mean([r[0] for r in cfg_results])
            mean_fs = {tau: np.mean([r[1][tau] for r in cfg_results]) for tau in FSCORE_THRESHOLDS}
            print(f"  → {cfg_name}  mean CD={mean_cd:.4f}  "
                  f"F@0.01={mean_fs[0.01]:.4f}  F@0.02={mean_fs[0.02]:.4f}  F@0.05={mean_fs[0.05]:.4f}\n")

    # Save CSV
    csv_path = os.path.join(sweep_dir, "results.csv")
    fieldnames = ["config", "sample", "chamfer"] + [f"f@{t}" for t in FSCORE_THRESHOLDS]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-sample results to {csv_path}")

    # Summary table
    print(f"\n{'config':<20} {'CD↓':>8} {'F@0.01↑':>9} {'F@0.02↑':>9} {'F@0.05↑':>9}  samples")
    print("-" * 70)
    for cfg_name in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg_name]
        if not cfg_rows:
            continue
        mean_cd    = np.mean([r["chamfer"]   for r in cfg_rows])
        mean_f01   = np.mean([r["f@0.01"]    for r in cfg_rows])
        mean_f02   = np.mean([r["f@0.02"]    for r in cfg_rows])
        mean_f05   = np.mean([r["f@0.05"]    for r in cfg_rows])
        n          = len(cfg_rows)
        print(f"{cfg_name:<20} {mean_cd:>8.4f} {mean_f01:>9.4f} {mean_f02:>9.4f} {mean_f05:>9.4f}  {n}")


if __name__ == "__main__":
    main()
