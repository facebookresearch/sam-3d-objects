"""Evaluation: compare a predicted mesh to a ground-truth mesh with Chamfer + F-score.

Public API: ``evaluate_pair(pred_path, gt_path, ...) -> dict`` runs the protocol
(independent normalization to [-1, 1] → area-weighted point sampling → optional
coarse-rotation + ICP alignment → Chamfer + F-score @ {0.005, 0.01, 0.02, 0.05})
and returns a flat dict of metrics.

Alignment is controlled by ``align``:
  - ``"none"``: no ICP. Use when you've handed in pre-aligned meshes.
  - ``"fixed"``: apply ``FIXED_ROTATION_DEFAULT`` then ICP. Cheap.
  - ``"grid"``: try all 24 cube rotations, keep the lowest post-ICP chamfer. Robust.

Default ``n=10_000``. PyTorch3D's ``knn_points`` is brute-force O(N·M) on CPU,
so N=1M hangs on the login node. N=1M paper-protocol mode comes with the
GPU-enabled runner.

CLI: ``python -m evaluation.run_eval --predicted PRED --ground_truth GT [--align grid]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.alignment import align_icp
from evaluation.mesh_io import load_mesh, normalize_mesh
from evaluation.metrics import chamfer, f_score, sample_points


def evaluate_pair(
    pred_path: str | Path,
    gt_path: str | Path,
    *,
    n: int = 10_000,
    seed: int = 0,
    align: str = "grid",
    device: str | torch.device | None = None,
) -> dict:
    """Run the evaluation protocol on one (predicted, GT) mesh pair.

    Steps:
      1. Load both meshes, normalize each to [-1, 1] independently.
      2. Uniformly sample ``n`` points from each surface (area-weighted, seeded).
      3. If ``align != "none"``, run ``align_icp(points_p, points_g, mode=align)``.
      4. Compute Chamfer + F-score @ {0.005, 0.01, 0.02, 0.05}.

    Returns a flat dict with keys ``chamfer``, ``f1@0.005``, ``f1@0.01``,
    ``f1@0.02``, ``f1@0.05``, plus ``n``, ``seed``, and ``align`` for provenance.
    """
    verts_p, faces_p = load_mesh(pred_path)
    verts_g, faces_g = load_mesh(gt_path)
    verts_p = normalize_mesh(verts_p)
    verts_g = normalize_mesh(verts_g)

    points_p = sample_points(verts_p, faces_p, n=n, seed=seed, device=device)
    points_g = sample_points(verts_g, faces_g, n=n, seed=seed, device=device)

    if align != "none":
        points_p = align_icp(points_p, points_g, mode=align)

    f1 = f_score(points_p, points_g)
    out: dict = {"chamfer": chamfer(points_p, points_g)}
    for tau, value in sorted(f1.items()):
        out[f"f1@{tau:g}"] = value
    out["n"] = n
    out["seed"] = seed
    out["align"] = align
    return out


def _format_report(result: dict) -> str:
    lines = [
        f"  n          : {result['n']}",
        f"  seed       : {result['seed']}",
        f"  align      : {result['align']}",
        f"  chamfer    : {result['chamfer']:.6f}",
        "  f_score    :",
    ]
    for key in sorted(k for k in result if k.startswith("f1@")):
        lines.append(f"    {key}  : {result[key]:.6f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predicted", required=True, type=Path, help="Path to predicted mesh.")
    parser.add_argument("--ground_truth", required=True, type=Path, help="Path to ground-truth mesh.")
    parser.add_argument("--n", type=int, default=10_000, help="Points sampled per mesh (default 10K, CPU-friendly).")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--align",
        choices=["none", "fixed", "grid"],
        default="grid",
        help="Coarse-rotation + ICP alignment mode (default: grid).",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON instead of human report.")
    args = parser.parse_args()

    result = evaluate_pair(args.predicted, args.ground_truth, n=args.n, seed=args.seed, align=args.align)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"predicted    : {args.predicted}")
        print(f"ground_truth : {args.ground_truth}")
        print(_format_report(result))


if __name__ == "__main__":
    main()
