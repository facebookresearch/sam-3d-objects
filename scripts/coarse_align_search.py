"""Grid-search over the 24 axis-aligned rotations as coarse ICP init, then save the
top-k aligned predictions as OBJs for visual inspection in Blender.

Run from repo root:
    python -m scripts.coarse_align_search \
        --predicted data/validation/coffee_cup_prediction.obj \
        --ground_truth data/validation/coffee_cup_gt.obj \
        --out outputs/eval_coarse_align --n 5000 --topk 5

Outputs (per `--out` dir):
    00_pred_norm.obj            normalized prediction, no rotation, no ICP
    00_gt_norm.obj              normalized ground truth
    00_pred_post_icp.obj        normalized prediction after ICP only (no coarse rot)
    rankK_rotNN_cdX.XXXX.obj    top-k by post-ICP chamfer (K = rank, NN = rotation idx)
    summary.csv                 all 24 rotations: pre-icp + post-icp chamfer
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import iterative_closest_point

from evaluation.alignment import cube_rotations
from evaluation.mesh_io import load_mesh, normalize_mesh, save_mesh
from evaluation.metrics import chamfer, sample_points


def label_rotation(R: np.ndarray) -> str:
    """e.g. 'x:+y y:-x z:+z' — which signed source axis each output axis maps to."""
    axes = ["x", "y", "z"]
    parts = []
    for i in range(3):
        col = int(np.argmax(np.abs(R[i])))
        sign = "+" if R[i, col] > 0 else "-"
        parts.append(f"{axes[i]}:{sign}{axes[col]}")
    return " ".join(parts)


def apply_to_verts(verts_t: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Apply pytorch3d SimilarityTransform to row-vector verts: v_out = s * v @ R + T."""
    return s * (verts_t @ R) + T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predicted", required=True, type=Path)
    p.add_argument("--ground_truth", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n", type=int, default=5000, help="Points sampled per mesh.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_iter", type=int, default=50, help="ICP iterations.")
    p.add_argument("--topk", type=int, default=5, help="How many top rotations to save as OBJ.")
    p.add_argument("--device", default=None, help="cpu, cuda, or auto.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    vp_raw, fp = load_mesh(args.predicted)
    vg_raw, fg = load_mesh(args.ground_truth)
    vp = normalize_mesh(vp_raw)
    vg = normalize_mesh(vg_raw)

    save_mesh(vp, fp, args.out / "00_pred_norm.obj")
    save_mesh(vg, fg, args.out / "00_gt_norm.obj")

    pts_p = sample_points(vp, fp, n=args.n, seed=args.seed, device=device)
    pts_g = sample_points(vg, fg, n=args.n, seed=args.seed, device=device)
    vp_t = torch.as_tensor(vp, dtype=torch.float32, device=device)

    # Baseline: ICP from identity (no coarse rotation).
    icp_id = iterative_closest_point(
        pts_p.unsqueeze(0), pts_g.unsqueeze(0),
        max_iterations=args.max_iter, estimate_scale=False,
    )
    cd_id = chamfer(icp_id.Xt[0], pts_g)
    v_id = apply_to_verts(vp_t, icp_id.RTs.R[0], icp_id.RTs.T[0], icp_id.RTs.s[0])
    save_mesh(v_id.cpu().numpy(), fp, args.out / f"00_pred_post_icp_cd{cd_id:.4f}.obj")
    print(f"baseline (no coarse rotation): post-ICP chamfer = {cd_id:.4f}")

    rotations = cube_rotations()
    print(f"trying {len(rotations)} cube rotations as ICP init...")
    results: list[dict] = []
    for idx, R in enumerate(rotations):
        R_t = torch.as_tensor(R, dtype=torch.float32, device=device)
        # row-vector convention: applying rotation R means p_new = p @ R.T
        pts_rot = pts_p @ R_t.T
        cd_pre = chamfer(pts_rot, pts_g)
        icp = iterative_closest_point(
            pts_rot.unsqueeze(0), pts_g.unsqueeze(0),
            max_iterations=args.max_iter, estimate_scale=False,
        )
        cd_post = chamfer(icp.Xt[0], pts_g)
        results.append({
            "idx": idx, "R": R, "R_t": R_t, "icp": icp,
            "cd_pre": cd_pre, "cd_post": cd_post,
        })
        print(f"  [{idx:2d}] {label_rotation(R):26s}  pre={cd_pre:.4f}  post={cd_post:.4f}")

    results.sort(key=lambda r: r["cd_post"])

    print("\nTop results by post-ICP chamfer:")
    for rank, r in enumerate(results[: args.topk]):
        print(
            f"  rank {rank}: rot {r['idx']:2d} ({label_rotation(r['R'])})  "
            f"pre={r['cd_pre']:.4f}  post={r['cd_post']:.4f}"
        )

    # Save top-k aligned meshes.
    for rank, r in enumerate(results[: args.topk]):
        v_rot = vp_t @ r["R_t"].T
        v_final = apply_to_verts(
            v_rot, r["icp"].RTs.R[0], r["icp"].RTs.T[0], r["icp"].RTs.s[0]
        )
        out_path = args.out / f"rank{rank}_rot{r['idx']:02d}_cd{r['cd_post']:.4f}.obj"
        save_mesh(v_final.cpu().numpy(), fp, out_path)
        print(f"  saved -> {out_path}")

    # Full summary as CSV.
    csv_path = args.out / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "label", "cd_pre", "cd_post"])
        for r in sorted(results, key=lambda r: r["idx"]):
            w.writerow([r["idx"], label_rotation(r["R"]), f"{r['cd_pre']:.6f}", f"{r['cd_post']:.6f}"])
    print(f"\nwrote summary -> {csv_path}")


if __name__ == "__main__":
    main()
