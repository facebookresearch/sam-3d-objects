"""
Aggregate eval over all sweep outputs.

Reads outputs/sweep/<config>/<sample>/pred_mesh.obj, computes
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
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "notebook"))
from eval_single import chamfer, f_score, load_gt_points, normalize, seed_everything
from evaluation.alignment import align_icp

# Must match sweep.py exactly
DATA = "data/Open3DHOI/data"

SAMPLES = [
    {"name": "coffee_cup", "dir": f"{DATA}/coffee cup/drinking_98"},
    {"name": "wrench",     "dir": f"{DATA}/wrench/wrench"},
    {"name": "chair",      "dir": f"{DATA}/chair/2398495"},
]

FSCORE_THRESHOLDS = (0.005, 0.01, 0.02, 0.05)

FAMILY_ORDER = {"ss": 1, "pose": 2, "depth": 3, "normal": 4}


def parse_config(name):
    """'ss_5.0' -> ('ss', 5.0); 'baseline' -> ('baseline', 0.0)."""
    if name == "baseline":
        return ("baseline", 0.0)
    fam, _, scale = name.partition("_")
    try:
        return (fam, float(scale))
    except ValueError:
        return (fam, 0.0)


def config_sort_key(name):
    fam, scale = parse_config(name)
    if fam == "baseline":
        return (0, 0.0)
    return (FAMILY_ORDER.get(fam, 9), scale)


def eval_one(pred_dir, gt_obj_path, device):
    """24-rotation grid ICP — matches eval_baseline_all.py protocol via evaluation.alignment."""
    pred = normalize(load_gt_points(os.path.join(pred_dir, "pred_mesh.obj"), device=device))
    gt   = normalize(load_gt_points(gt_obj_path, device=device))
    pred_icp = align_icp(pred, gt, mode="grid")
    cd = float(chamfer(pred_icp, gt))
    fs = f_score(pred_icp, gt, thresholds=FSCORE_THRESHOLDS)
    return cd, fs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default="outputs/sweep")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Only evaluate these configs (e.g. --configs ss_0.1 ss_0.2)")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    device    = args.device
    seed_everything()
    # Discover configs from directory
    configs = sorted([
        d for d in os.listdir(sweep_dir)
        if os.path.isdir(os.path.join(sweep_dir, d)) and d != "results.csv"
    ])

    if args.configs:
        configs = [c for c in configs if c in args.configs]

    if not configs:
        print(f"No configs found in {sweep_dir}")
        sys.exit(1)

    print(f"Found {len(configs)} configs: {configs}\n")

    rows = []  # list of dicts for CSV

    for cfg_name in configs:
        cfg_results = []
        for sample in SAMPLES:
            pred_dir  = os.path.join(sweep_dir, cfg_name, sample["name"])
            pred_path = os.path.join(pred_dir, "pred_mesh.obj")
            gt_path   = os.path.join(sample["dir"], "object_mesh.obj")

            if not os.path.exists(pred_path):
                print(f"  SKIP {cfg_name}/{sample['name']} — no pred_mesh.obj")
                continue

            print(f"  {cfg_name} / {sample['name']} ...", end=" ", flush=True)
            cd, fs = eval_one(pred_dir, gt_path, device)
            print(f"CD={cd:.4f}  F@0.02={fs[0.02]:.4f}")

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

    # mean CD per config
    mean_cd = {}
    for cfg_name in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg_name]
        if cfg_rows:
            mean_cd[cfg_name] = np.mean([r["chamfer"] for r in cfg_rows])

    base_cd = mean_cd.get("baseline")

    # Summary table (sorted by family then numeric scale)
    print(f"\n{'config':<20} {'CD↓':>8} {'F@0.01↑':>9} {'F@0.02↑':>9} {'F@0.05↑':>9}  samples")
    print("-" * 70)
    for cfg_name in sorted(mean_cd, key=config_sort_key):
        cfg_rows = [r for r in rows if r["config"] == cfg_name]
        m_cd  = np.mean([r["chamfer"] for r in cfg_rows])
        m_f01 = np.mean([r["f@0.01"]  for r in cfg_rows])
        m_f02 = np.mean([r["f@0.02"]  for r in cfg_rows])
        m_f05 = np.mean([r["f@0.05"]  for r in cfg_rows])
        print(f"{cfg_name:<20} {m_cd:>8.4f} {m_f01:>9.4f} {m_f02:>9.4f} {m_f05:>9.4f}  {len(cfg_rows)}")

    # ── RANGE ANALYSIS ──────────────────────────────────────────────────────
    # For each guidance family: CD vs scale, Δ vs baseline, sweet spot.
    print(f"\n\n{'='*64}")
    print("RANGE ANALYSIS  (Δ = mean CD − baseline; negative = improvement)")
    if base_cd is not None:
        print(f"baseline mean CD = {base_cd:.4f}")
    print("=" * 64)

    fam_entries = defaultdict(list)  # family -> [(scale, cfg, mean_cd)]
    for cfg_name, m in mean_cd.items():
        fam, scale = parse_config(cfg_name)
        if fam in FAMILY_ORDER:
            fam_entries[fam].append((scale, cfg_name, m))

    for fam in sorted(fam_entries, key=lambda f: FAMILY_ORDER[f]):
        entries = sorted(fam_entries[fam])  # by scale
        print(f"\n[{fam}]   scale      CD      Δ vs base   trend")
        print(f"  {'-'*46}")
        prev = base_cd
        for scale, cfg_name, m in entries:
            if base_cd is not None:
                delta = m - base_cd
                dstr = f"{'+' if delta >= 0 else ''}{delta:.4f}"
                mark = "improve" if delta < -1e-4 else ("worse" if delta > 1e-4 else "~same")
            else:
                dstr, mark = "n/a", ""
            arrow = ""
            if prev is not None:
                arrow = "down" if m < prev - 1e-4 else ("up" if m > prev + 1e-4 else "flat")
            print(f"  {scale:>8.3f}  {m:>8.4f}   {dstr:>9}   {mark:<7} {arrow}")
            prev = m

        best_scale, best_cfg, best_m = min(entries, key=lambda e: e[2])
        if base_cd is not None:
            verdict = (f"BEST {best_cfg} (CD={best_m:.4f}, Δ={best_m - base_cd:+.4f})"
                       if best_m < base_cd - 1e-4
                       else f"no config beats baseline (best {best_cfg} CD={best_m:.4f})")
        else:
            verdict = f"BEST {best_cfg} (CD={best_m:.4f})"
        print(f"  -> {verdict}")

    print(f"\n{'='*64}")
    print("Read the 'trend' column: CD should drop then rise — pick the scale")
    print("at the minimum. If still dropping at the max scale, range is too low.")
    print("=" * 64)


if __name__ == "__main__":
    main()
