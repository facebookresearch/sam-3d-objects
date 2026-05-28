"""
Collect top-N worst and bottom-N best instances from results.csv into inspection folders.

Creates:
  outputs/inspection/worst/<rank>_<category>_<instance>/
  outputs/inspection/best/<rank>_<category>_<instance>/

Each folder contains: gt.obj, pred.obj, pose_params.pt, image.jpg, mask.png

Usage:
    python scripts/collect_top30_worst.py
    python scripts/collect_top30_worst.py --worst 30 --best 10
    python scripts/collect_top30_worst.py --csv outputs/baseline_all/results.csv
"""

import argparse
import csv
import os
import shutil

PRED_ROOT = "outputs/baseline_all"
DATA_ROOT = "data/Open3DHOI/data"
OUT_DIR   = "outputs/inspection"


def collect(rows, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Collecting {len(rows)} {label} instances → {out_dir}/\n")

    for rank, row in enumerate(rows, start=1):
        cat  = row["category"]
        inst = row["instance"]
        cd   = float(row["chamfer"])

        folder_name = f"{rank:03d}_{cat}_{inst}".replace(" ", "_").replace("/", "_")
        dest = os.path.join(out_dir, folder_name)
        os.makedirs(dest, exist_ok=True)

        files = {
            os.path.join(DATA_ROOT, cat, inst, "object_mesh.obj"): "gt.obj",
            os.path.join(PRED_ROOT, cat, inst, "pred_mesh.obj"):   "pred.obj",
            os.path.join(PRED_ROOT, cat, inst, "pose_params.pt"):  "pose_params.pt",
            os.path.join(DATA_ROOT, cat, inst, "image.jpg"):       "image.jpg",
            os.path.join(DATA_ROOT, cat, inst, "obj_mask.png"):    "mask.png",
        }

        copied, missing = [], []
        for src, dst_name in files.items():
            dst = os.path.join(dest, dst_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied.append(dst_name)
            else:
                missing.append(dst_name)

        status = f"CD={cd:.4f}  copied={copied}"
        if missing:
            status += f"  MISSING={missing}"
        print(f"[{rank:3d}] {cat}/{inst}  {status}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default=os.path.join(PRED_ROOT, "results.csv"))
    parser.add_argument("--worst", type=int, default=30)
    parser.add_argument("--best",  type=int, default=10)
    parser.add_argument("--out",   default=OUT_DIR)
    args = parser.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    collect(rows[:args.worst],        os.path.join(args.out, "worst"), "worst")
    collect(rows[-args.best:][::-1],  os.path.join(args.out, "best"),  "best")

    print(f"Done. Inspect at {args.out}/worst/ and {args.out}/best/")


if __name__ == "__main__":
    main()
