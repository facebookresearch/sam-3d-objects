"""
Run baseline (no guidance) inference on selected Open3DHOI instances.

Reads categories from categories.txt (uncommented lines = included;
all-commented means include all listed categories).
Caps at --cap instances per category, sorted alphabetically by instance id.
Skips instances where pred_mesh.obj already exists.
Outputs land in outputs/baseline_all/<category>/<instance_id>/.

Usage:
    python baseline_all.py                          # all selected, no split
    python baseline_all.py --offset 0 --limit 533  # first half
    python baseline_all.py --offset 533             # second half
    python baseline_all.py --cap 50 --tag hf --seed 42
"""

import argparse
import os
import sys

import torch

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "notebook"))
from inference import Inference, load_image, load_mask
from main import build_guidance

DATA_ROOT      = "data/Open3DHOI/data"
OUT_ROOT       = "outputs/baseline_all"
CATEGORIES_TXT = "categories.txt"


def load_categories():
    """
    Return list of selected category names from categories.txt.
    Uncommented lines are explicitly selected.
    If nothing is uncommented, fall back to all listed (commented) categories.
    """
    explicit, listed = [], []
    with open(CATEGORIES_TXT) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("# Pick") or line.startswith("# Format") or line.startswith("# Instance"):
                continue
            if line.startswith("#"):
                cat = line.lstrip("#").split("|")[0].strip()
                if cat:
                    listed.append(cat)
            else:
                cat = line.split("|")[0].strip()
                if cat:
                    explicit.append(cat)
    return explicit if explicit else listed


def discover_instances(categories, cap):
    """Return sorted list of (category, instance_id, instance_dir), capped per category."""
    instances = []
    for cat in categories:
        cat_dir = os.path.join(DATA_ROOT, cat)
        if not os.path.isdir(cat_dir):
            continue
        insts = sorted([
            i for i in os.listdir(cat_dir)
            if os.path.isdir(os.path.join(cat_dir, i)) and
               os.path.exists(os.path.join(cat_dir, i, "image.jpg")) and
               os.path.exists(os.path.join(cat_dir, i, "obj_mask.png"))
        ])
        for inst in insts[:cap]:
            instances.append((cat, inst, os.path.join(cat_dir, inst)))
    return instances


def load_instances_file(path):
    """Load explicit cat/instance pairs from a text file (one per line)."""
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("/", 1)
            if len(parts) == 2:
                cat, inst = parts
                inst_dir = os.path.join(DATA_ROOT, cat, inst)
                instances.append((cat, inst, inst_dir))
    return instances


def run(tag, seed, cap, offset, limit, instances_file=None):
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    print(f"Loading model from {config_path} ...")
    inference = Inference(config_path, compile=False)
    print("Model loaded.\n")

    if instances_file:
        subset = load_instances_file(instances_file)
        print(f"Explicit instances from {instances_file}: {len(subset)}\n")
        end = len(subset)
    else:
        categories = load_categories()
        all_instances = discover_instances(categories, cap)
        print(f"Categories: {len(categories)}  Total instances (cap={cap}): {len(all_instances)}")
        subset = all_instances[offset : offset + limit if limit else None]
        end = offset + len(subset)
        print(f"This job: [{offset+1}:{end}] = {len(subset)} instances.\n")

    done = skipped = failed = 0

    for i, (cat, inst, inst_dir) in enumerate(subset):
        global_idx = i + 1 if instances_file else offset + i + 1
        out_dir = os.path.join(OUT_ROOT, cat, inst)
        pred_mesh_path  = os.path.join(out_dir, "pred_mesh.obj")
        pose_params_path = os.path.join(out_dir, "pose_params.pt")

        if os.path.exists(pred_mesh_path) and os.path.exists(pose_params_path):
            skipped += 1
            print(f"[{global_idx}/{end}] SKIP  {cat}/{inst}")
            continue

        print(f"[{global_idx}/{end}] RUN   {cat}/{inst}")

        try:
            image = load_image(os.path.join(inst_dir, "image.jpg"))
            mask  = load_mask(os.path.join(inst_dir, "obj_mask.png"))

            guidance = build_guidance(
                mask_path=os.path.join(inst_dir, "obj_mask.png"),
                ss_guidance_scale=None,
                pose_guidance_scale=None,
                depth_guidance_scale=None,
                normal_guidance_scale=None,
                depth_map=None,
                steps_prefix=None,
                w_centroid=1.0,
                w_size=1.0,
            )

            output = inference(image, mask, seed=seed, steps_prefix=None, guidance=guidance)

            os.makedirs(out_dir, exist_ok=True)

            if "gs" in output:
                np.save(os.path.join(out_dir, "pred_points.npy"),
                        output["gs"].get_xyz.detach().cpu().numpy())

            if "mesh" in output:
                mesh = output["mesh"][0]
                if mesh.success:
                    tm = trimesh.Trimesh(
                        vertices=mesh.vertices.cpu().numpy(),
                        faces=mesh.faces.cpu().numpy(),
                    )
                    tm.export(pred_mesh_path)
                    print(f"         -> saved to {out_dir}")
                else:
                    print(f"         -> mesh extraction failed")

            # Save MoGe pose + intrinsics for camera-space evaluation
            pose_params = {
                k: output[k].cpu() if hasattr(output.get(k), "cpu") else output.get(k)
                for k in ("rotation", "translation", "scale", "intrinsics")
                if k in output
            }
            if pose_params:
                torch.save(pose_params, pose_params_path)

            done += 1

        except Exception as e:
            failed += 1
            print(f"         -> ERROR: {e}")

    print(f"\nDone: {done}  Skipped: {skipped}  Failed: {failed}")
    print("baseline_all complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",    default="hf")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--cap",    type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--instances", default=None,
                        help="text file with cat/instance per line to run explicitly")
    args = parser.parse_args()
    run(args.tag, args.seed, args.cap, args.offset, args.limit, args.instances)


if __name__ == "__main__":
    main()
