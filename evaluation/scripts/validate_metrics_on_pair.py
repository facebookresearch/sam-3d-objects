"""Validate that the evaluation/ metrics behave correctly on a (predicted, GT) mesh pair.

Five checks:
  1. Load + inspect both meshes (pre-normalization).
  2. evaluate_pair(PRED, GT) on the real pair.
  3. evaluate_pair(GT, GT) identity baseline.
  4. Perturbed GT sweep at σ ∈ {0.001, 0.005, 0.01, 0.05}. Noise is applied AFTER
     normalization, so σ is in [-1, 1] cube units (comparable to test_sanity.py).
  5. Bracket PRED's Chamfer against the σ sweep.

Use this on any (predicted, GT) pair to sanity-check the eval pipeline end-to-end
before reading absolute numbers. Identity should be exact; the σ sweep should be
strictly monotone in Chamfer. Wildly off either of those means the input pair has
a problem (frame mismatch, broken normalization) — not that the metric is wrong.

Run:
    python -m evaluation.scripts.validate_metrics_on_pair \\
        --predicted    path/to/pred.obj \\
        --ground_truth path/to/gt.obj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support direct file invocation (python path/to/script.py) in addition to `python -m`.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.mesh_io import load_mesh, normalize_mesh, save_mesh
from evaluation.perturbations import add_vertex_noise
from evaluation.run_eval import evaluate_pair

SIGMAS = (0.001, 0.005, 0.01, 0.05)
METRIC_KEYS = ("chamfer", "f1@0.005", "f1@0.01", "f1@0.02", "f1@0.05")


def inspect(name: str, path: str) -> None:
    verts, faces = load_mesh(path)
    bbmin = verts.min(axis=0)
    bbmax = verts.max(axis=0)
    extent = bbmax - bbmin
    print(f"{name} : {path}")
    print(f"       V={verts.shape[0]:>8}  F={faces.shape[0]:>8}")
    print(f"       bbox_min = ({bbmin[0]:+.4f}, {bbmin[1]:+.4f}, {bbmin[2]:+.4f})")
    print(f"       bbox_max = ({bbmax[0]:+.4f}, {bbmax[1]:+.4f}, {bbmax[2]:+.4f})")
    print(f"       extent   = ({extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f})")


def fmt_one_line(result: dict) -> str:
    return "  ".join(f"{k}={result[k]:.6f}" for k in METRIC_KEYS)


def banner(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predicted", required=True, type=Path, help="Path to predicted mesh.")
    parser.add_argument("--ground_truth", required=True, type=Path, help="Path to ground-truth mesh.")
    parser.add_argument("--n", type=int, default=10_000, help="Points sampled per mesh (MVP default 10K).")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed; also seeds perturbation noise.")
    parser.add_argument(
        "--tmp_dir",
        type=Path,
        default=Path("/tmp"),
        help="Directory for the perturbed GT meshes written during the σ sweep.",
    )
    args = parser.parse_args()

    pred_path = str(args.predicted)
    gt_path = str(args.ground_truth)

    banner("Check 1: load & inspect (pre-normalization)")
    inspect("PRED", pred_path)
    print()
    inspect("GT  ", gt_path)

    banner("Check 2: evaluate_pair(PRED, GT)")
    pred_result = evaluate_pair(pred_path, gt_path, n=args.n, seed=args.seed)
    print(fmt_one_line(pred_result))

    banner("Check 3: identity baseline — evaluate_pair(GT, GT)")
    id_result = evaluate_pair(gt_path, gt_path, n=args.n, seed=args.seed)
    print(fmt_one_line(id_result))

    banner("Check 4: perturbed GT sweep (vertex noise on normalized GT, vs original GT)")
    # Perturb post-normalization so σ is in [-1, 1] cube units — matches test_sanity.py.
    # evaluate_pair will re-normalize the saved perturbed mesh; for small σ this is a
    # near-no-op (bbox stays ≈ 2.0). At σ=0.05 the bbox grows ~10% and re-norm shrinks
    # by the inverse — small artifact, acceptable at this resolution.
    gt_verts, gt_faces = load_mesh(gt_path)
    gt_verts_norm = normalize_mesh(gt_verts)
    header_cols = ["sigma"] + list(METRIC_KEYS)
    widths = [8, 12, 10, 10, 10, 10]
    print("  " + " | ".join(f"{c:>{w}}" for c, w in zip(header_cols, widths)))
    print("  " + " | ".join("-" * w for w in widths))
    sweep: list[tuple[float, dict]] = []
    for sigma in SIGMAS:
        pert_verts = add_vertex_noise(gt_verts_norm, sigma=sigma, seed=args.seed)
        pert_path = args.tmp_dir / f"gt_pert_{sigma:g}.obj"
        save_mesh(pert_verts, gt_faces, pert_path)
        r = evaluate_pair(str(pert_path), gt_path, n=args.n, seed=args.seed)
        sweep.append((sigma, r))
        row = [f"{sigma:.4f}", f"{r['chamfer']:.6e}"] + [f"{r[k]:.6f}" for k in METRIC_KEYS[1:]]
        print("  " + " | ".join(f"{c:>{w}}" for c, w in zip(row, widths)))

    banner("Check 5: interpretation")
    pred_cham = pred_result["chamfer"]
    sigmas = [s for s, _ in sweep]
    chams = [r["chamfer"] for _, r in sweep]
    if not all(chams[i] <= chams[i + 1] for i in range(len(chams) - 1)):
        print(f"WARNING: σ sweep chamfers are not monotone: {chams}. Bracketing may be unreliable.")

    if pred_cham < chams[0]:
        print(
            f"PRED chamfer = {pred_cham:.6f} is BELOW σ={sigmas[0]} (chamfer {chams[0]:.6f}) — "
            f"prediction is closer to GT than the smallest tested perturbation."
        )
    elif pred_cham > chams[-1]:
        print(
            f"PRED chamfer = {pred_cham:.6f} is ABOVE σ={sigmas[-1]} (chamfer {chams[-1]:.6f}) — "
            f"prediction is worse than σ={sigmas[-1]} vertex noise on GT."
        )
    else:
        for i in range(len(sigmas) - 1):
            if chams[i] <= pred_cham <= chams[i + 1]:
                print(
                    f"PRED vs GT lands BETWEEN σ={sigmas[i]} and σ={sigmas[i + 1]} "
                    f"(chamfer {pred_cham:.6f} ∈ [{chams[i]:.6f}, {chams[i + 1]:.6f}])."
                )
                break


if __name__ == "__main__":
    main()
