"""
Scale sweep for Stage 1 guidance over a fixed set of samples.

Loads the model once, then runs every (sample, config) combination.
Outputs land in outputs/sweep/<config_name>/<sample_name>/.

Usage:
    python sweep.py
    python sweep.py --tag hf --seed 42
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

sys.path.append("notebook")
from inference import Inference, load_image, load_mask
from main import build_guidance

# ── Samples ──────────────────────────────────────────────────────────────────

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

# ── Sweep configs ─────────────────────────────────────────────────────────────
# Independent sweep: vary one param at a time.

CONFIGS = [
    # baseline
    {"name": "baseline",       "ss": None,  "pose": None, "depth": None},
    # shape only
    {"name": "ss_0.1",         "ss": 0.1,   "pose": None, "depth": None},
    {"name": "ss_0.2",         "ss": 0.2,   "pose": None, "depth": None},
    {"name": "ss_0.5",         "ss": 0.5,   "pose": None, "depth": None},
    # pose only
    {"name": "pose_0.05",      "ss": None,  "pose": 0.05, "depth": None},
    {"name": "pose_0.1",       "ss": None,  "pose": 0.1,  "depth": None},
    {"name": "pose_0.2",       "ss": None,  "pose": 0.2,  "depth": None},
    {"name": "pose_0.5",       "ss": None,  "pose": 0.5,  "depth": None},
    # depth only
    {"name": "depth_0.1",      "ss": None,  "pose": None, "depth": 0.1},
    {"name": "depth_0.2",      "ss": None,  "pose": None, "depth": 0.2},
    {"name": "depth_0.5",      "ss": None,  "pose": None, "depth": 0.5},
]

# ─────────────────────────────────────────────────────────────────────────────


def run_sweep(tag: str, seed: int, n_samples: int = None):
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    print(f"Loading model from {config_path} ...")
    inference = Inference(config_path, compile=False)
    print("Model loaded.\n")

    samples = SAMPLES[:n_samples] if n_samples else SAMPLES
    total = len(samples) * len(CONFIGS)
    done = 0

    for cfg in CONFIGS:
        for sample in samples:
            done += 1
            out_dir = os.path.join("outputs", "sweep", cfg["name"], sample["name"])

            # Skip if already done
            if os.path.exists(os.path.join(out_dir, "pred_points.npy")):
                print(f"[{done}/{total}] SKIP {cfg['name']} / {sample['name']} (exists)")
                continue

            print(f"[{done}/{total}] RUN  {cfg['name']} / {sample['name']}")

            image   = load_image(os.path.join(sample["dir"], "image.jpg"))
            mask    = load_mask(os.path.join(sample["dir"], "obj_mask.png"))
            depth   = os.path.join(sample["dir"], "depth.npy") if cfg["depth"] else None
            prefix  = os.path.join("sweep", cfg["name"], sample["name"])

            guidance = build_guidance(
                mask_path=os.path.join(sample["dir"], "obj_mask.png"),
                ss_guidance_scale=cfg["ss"],
                pose_guidance_scale=cfg["pose"],
                depth_guidance_scale=cfg["depth"],
                depth_map=depth,
                steps_prefix=prefix,
                w_centroid=1.0,
                w_size=1.0,
            )

            output = inference(
                image, mask,
                seed=seed,
                steps_prefix=prefix,
                guidance=guidance,
            )

            os.makedirs(out_dir, exist_ok=True)
            if "gs" in output:
                np.save(os.path.join(out_dir, "pred_points.npy"),
                        output["gs"].get_xyz.detach().cpu().numpy())
                output["gs"].save_ply(os.path.join(out_dir, "gaussian.ply"))
            if "mesh" in output:
                mesh = output["mesh"][0]
                if mesh.success:
                    tm = trimesh.Trimesh(
                        vertices=mesh.vertices.cpu().numpy(),
                        faces=mesh.faces.cpu().numpy(),
                    )
                    tm.export(os.path.join(out_dir, "pred_mesh.obj"))

            print(f"         → saved to {out_dir}")

    print("\nSweep complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",       default="hf")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Run only the first N samples (default: all 10)")
    args = parser.parse_args()
    run_sweep(args.tag, args.seed, args.n_samples)


if __name__ == "__main__":
    main()
