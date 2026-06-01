"""
Scale sweep for Stage 1 guidance over a fixed set of samples.

Loads the model once, then runs every (sample, config) combination.
Outputs land in outputs/sweep/<config_name>/<cat>/<inst>/.

Usage:
    python sweep.py                              # hardcoded SAMPLES × CONFIGS
    python sweep.py --instances worst_30.txt     # file-driven instances × CONFIGS
    python sweep.py --instances worst_30.txt --configs ss_1.0 ss_5.0 ss5_pose005
    python sweep.py --tag hf --seed 42
"""

import argparse
import os
import sys

import numpy as np
import torch
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "notebook"))
from inference import Inference, load_image, load_mask
from main import build_guidance

# ── Hardcoded samples (used when --instances not provided) ───────────────────

DATA = "data/Open3DHOI/data"

SAMPLES = [
    {"name": "wrench", "dir": f"{DATA}/wrench/wrench"},
    {"name": "chair",  "dir": f"{DATA}/chair/2398495"},
]

# ── Sweep configs ─────────────────────────────────────────────────────────────

CONFIGS = {
    # baseline — no guidance
    "baseline":      {"ss": None,  "pose": None, "depth": None, "normal": None},
    # shape guidance only (sweet spot ~2-3 on dev wrench; ss>=10 destructive)
    "ss_0.1":        {"ss": 0.1,   "pose": None, "depth": None, "normal": None},
    "ss_0.5":        {"ss": 0.5,   "pose": None, "depth": None, "normal": None},
    "ss_1.0":        {"ss": 1.0,   "pose": None, "depth": None, "normal": None},
    "ss_2.0":        {"ss": 2.0,   "pose": None, "depth": None, "normal": None},
    "ss_3.0":        {"ss": 3.0,   "pose": None, "depth": None, "normal": None},
    "ss_5.0":        {"ss": 5.0,   "pose": None, "depth": None, "normal": None},
    "ss_10.0":       {"ss": 10.0,  "pose": None, "depth": None, "normal": None},
    # pose guidance only (no effect on normalized+ICP CD — eval via silhouette IoU)
    "pose_0.05":     {"ss": None,  "pose": 0.05, "depth": None, "normal": None},
    "pose_0.2":      {"ss": None,  "pose": 0.2,  "depth": None, "normal": None},
    # depth guidance only
    "depth_2.0":     {"ss": None,  "pose": None, "depth": 2.0,  "normal": None},
    "depth_5.0":     {"ss": None,  "pose": None, "depth": 5.0,  "normal": None},
    # normal guidance only
    "normal_2.0":    {"ss": None,  "pose": None, "depth": None, "normal": 2.0},
    "normal_5.0":    {"ss": None,  "pose": None, "depth": None, "normal": 5.0},
    # combos
    "ss5_pose005":   {"ss": 5.0,   "pose": 0.05, "depth": None, "normal": None},
    "ss5_depth5":    {"ss": 5.0,   "pose": None, "depth": 5.0,  "normal": None},
}

DEFAULT_CONFIGS = ["ss_1.0", "ss_2.0", "ss_3.0", "ss_5.0",
                   "pose_0.05", "pose_0.2",
                   "depth_2.0", "depth_5.0",
                   "normal_2.0", "normal_5.0"]

# ─────────────────────────────────────────────────────────────────────────────


def load_instances(path):
    """Read cat/inst lines from a text file, return list of sample dicts."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("/", 1)
            if len(parts) == 2:
                cat, inst = parts
                samples.append({
                    "name": f"{cat}/{inst}",
                    "cat":  cat,
                    "inst": inst,
                    "dir":  os.path.join(DATA, cat, inst),
                })
    return samples


def run_sweep(tag: str, seed: int, samples, config_names: list):
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    print(f"Loading model from {config_path} ...")
    inference = Inference(config_path, compile=False)
    print("Model loaded.\n")

    configs = [(name, CONFIGS[name]) for name in config_names]
    total = len(samples) * len(configs)
    done = 0

    for cfg_name, cfg in configs:
        for sample in samples:
            done += 1
            cat  = sample.get("cat",  sample["name"])
            inst = sample.get("inst", sample["name"])
            out_dir = os.path.join("outputs", "sweep", cfg_name, cat, inst)

            pred_mesh_path  = os.path.join(out_dir, "pred_mesh.obj")
            pose_params_path = os.path.join(out_dir, "pose_params.pt")

            if os.path.exists(pred_mesh_path) and os.path.exists(pose_params_path):
                print(f"[{done}/{total}] SKIP {cfg_name} / {sample['name']} (exists)")
                continue

            print(f"[{done}/{total}] RUN  {cfg_name} / {sample['name']}")

            inst_dir = sample["dir"]
            image = load_image(os.path.join(inst_dir, "image.jpg"))
            mask  = load_mask(os.path.join(inst_dir, "obj_mask.png"))
            depth_map = os.path.join(inst_dir, "depth.npy") if (cfg["depth"] or cfg["normal"]) else None

            prefix = os.path.join("sweep", cfg_name, cat, inst)

            guidance = build_guidance(
                mask_path=os.path.join(inst_dir, "obj_mask.png"),
                ss_guidance_scale=cfg["ss"],
                pose_guidance_scale=cfg["pose"],
                depth_guidance_scale=cfg["depth"],
                normal_guidance_scale=cfg["normal"],
                depth_map=depth_map,
                steps_prefix=prefix,
                w_centroid=1.0,
                w_size=1.0,
            )

            output = inference(image, mask, seed=seed, steps_prefix=prefix, guidance=guidance)

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

            pose_params = {
                k: output[k].cpu() if hasattr(output.get(k), "cpu") else output.get(k)
                for k in ("rotation", "translation", "scale", "intrinsics")
                if k in output
            }
            if pose_params:
                torch.save(pose_params, pose_params_path)

            print(f"         → saved to {out_dir}")

    print("\nSweep complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",       default="hf")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--instances", default=None,
                        help="text file with cat/instance lines; uses hardcoded SAMPLES if omitted")
    parser.add_argument("--configs",   nargs="+", default=DEFAULT_CONFIGS,
                        help=f"config names to run (default: {DEFAULT_CONFIGS}); "
                             f"available: {sorted(CONFIGS)}")
    args = parser.parse_args()

    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        parser.error(f"Unknown config names: {unknown}. Available: {sorted(CONFIGS)}")

    samples = load_instances(args.instances) if args.instances else SAMPLES
    print(f"Instances: {len(samples)}  Configs: {args.configs}\n")

    run_sweep(args.tag, args.seed, samples, args.configs)


if __name__ == "__main__":
    main()
