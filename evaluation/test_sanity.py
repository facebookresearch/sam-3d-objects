"""Synthetic sanity check: validate that Chamfer + F-score behave as expected on a known mesh.

Loads an icosphere, applies a sweep of controlled perturbations (vertex noise σ and
translation), and asserts the hard behaviors from the eval plan §"GT + noise":

  - Identity: same mesh, same sampling seed → Chamfer ≈ 0, F1@0.01 ≈ 1.0
  - Monotonicity (Chamfer): strictly increases as σ ∈ {0, 0.001, 0.005, 0.01, 0.05, 0.1}
  - Monotonicity (F-score): F1@0.01 strictly decreases across σ ∈ {0.001, 0.01, 0.05}
    (widely spaced σ vs. τ keep this robust to sampling noise — see assertion comment)
  - Symmetry: Chamfer(A, B) == Chamfer(B, A) to float precision
  - Translation sensitivity: translating by 0.1 hurts Chamfer measurably (>1e-3) but
    not catastrophically (<0.5)

No ICP (we control alignment by construction). Operates in memory on the primitives
in ``metrics``, so a regression in ``chamfer`` / ``f_score`` / ``sample_points`` is
caught here without going through the file-loading paths in ``run_eval.evaluate_pair``.

Run: ``python -m evaluation.test_sanity`` from the repo root once the conda env is
healthy. (As of 2026-05-18 the ``/scratch-shared`` env's stdlib is corrupted; see
``evaluation/README.md`` § "Verifying your install" for recovery.)
Exits 0 on success, raises AssertionError on failure.
"""

from __future__ import annotations

import numpy as np
import trimesh

from evaluation.mesh_io import normalize_mesh
from evaluation.metrics import chamfer, f_score, sample_points
from evaluation.perturbations import add_vertex_noise, translate

N = 10_000
SEED = 0
SIGMAS = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)


def main() -> None:
    base = trimesh.creation.icosphere(subdivisions=3)
    verts = normalize_mesh(np.asarray(base.vertices))
    faces = np.asarray(base.faces)
    print(f"icosphere: V={verts.shape[0]}, F={faces.shape[0]}, N={N}, seed={SEED}")

    points_gt = sample_points(verts, faces, n=N, seed=SEED)

    # 1. Identity — same mesh, same seed → identical sample → Chamfer = 0, F1@0.01 = 1.0.
    points_id = sample_points(verts, faces, n=N, seed=SEED)
    cham_id = chamfer(points_id, points_gt)
    f1_id = f_score(points_id, points_gt)[0.01]
    print(f"\nidentity     : chamfer={cham_id:.3e}  f1@0.01={f1_id:.6f}")
    assert cham_id < 1e-4, f"identity chamfer {cham_id:.3e} not near zero"
    assert f1_id > 0.999, f"identity F1@0.01 {f1_id:.6f} not near 1.0"

    # 2. Symmetry — two different samples of the same mesh; Chamfer should be symmetric.
    points_b = sample_points(verts, faces, n=N, seed=SEED + 1)
    cham_ab = chamfer(points_gt, points_b)
    cham_ba = chamfer(points_b, points_gt)
    print(f"symmetry     : chamfer(A,B)={cham_ab:.6e}  chamfer(B,A)={cham_ba:.6e}  |Δ|={abs(cham_ab - cham_ba):.2e}")
    assert abs(cham_ab - cham_ba) < 1e-6, f"chamfer not symmetric: |Δ|={abs(cham_ab - cham_ba):.3e}"

    # 3. σ sweep — monotone in noise (Chamfer up, F1 down).
    print("\nσ sweep:")
    print(f"  {'sigma':>8} | {'chamfer':>12} | {'f1@0.01':>10}")
    print(f"  {'-' * 8} | {'-' * 12} | {'-' * 10}")
    chamfers = []
    f1s_at_01: dict[float, float] = {}
    for sigma in SIGMAS:
        verts_noisy = add_vertex_noise(verts, sigma=sigma, seed=SEED)
        points_noisy = sample_points(verts_noisy, faces, n=N, seed=SEED)
        c = chamfer(points_noisy, points_gt)
        f1 = f_score(points_noisy, points_gt)[0.01]
        chamfers.append(c)
        f1s_at_01[sigma] = f1
        print(f"  {sigma:>8.4f} | {c:>12.6e} | {f1:>10.6f}")
    for s_prev, s_curr, c_prev, c_curr in zip(SIGMAS, SIGMAS[1:], chamfers, chamfers[1:]):
        assert c_curr > c_prev, (
            f"chamfer not monotone: σ={s_prev}→{s_curr}, chamfer {c_prev:.4e}→{c_curr:.4e}"
        )

    # F1@0.01 monotone across {0.001, 0.01, 0.05}. Non-adjacent steps deliberately —
    # at σ=0.001 (σ « τ) F1 ≈ 1.0; at σ=0.01 (σ ≈ τ) F1 is in transition; at σ=0.05
    # (σ » τ) F1 is small. Wide gaps absorb sampling noise that an adjacent-step
    # assertion would trip on at N=10K.
    f1_lo, f1_mid, f1_hi = f1s_at_01[0.001], f1s_at_01[0.01], f1s_at_01[0.05]
    assert f1_lo > f1_mid > f1_hi, (
        f"F1@0.01 not monotone over σ∈{{0.001,0.01,0.05}}: {f1_lo:.4f} > {f1_mid:.4f} > {f1_hi:.4f}"
    )

    # 4. Translation sensitivity — bounded by (1e-3, 0.5).
    verts_t = translate(verts, t=(0.1, 0.0, 0.0))
    points_t = sample_points(verts_t, faces, n=N, seed=SEED)
    cham_t = chamfer(points_t, points_gt)
    print(f"\ntranslate 0.1: chamfer={cham_t:.6e}")
    assert 1e-3 < cham_t < 0.5, f"translation chamfer {cham_t:.3e} outside (1e-3, 0.5)"

    print("\nall sanity assertions passed.")


if __name__ == "__main__":
    main()
