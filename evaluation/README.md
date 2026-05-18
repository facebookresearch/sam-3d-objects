# Evaluation Pipeline (MVP)

A standalone toolkit for evaluating predicted 3D meshes against ground-truth meshes.
Compares two meshes with **Chamfer distance** and **F-score** at thresholds
`{0.005, 0.01, 0.02, 0.05}`. Modeled on the SAM 3D paper §D.3.1 protocol, scoped down
for the MVP (see *What's not yet included* below).

This pipeline is independent of the `sam3d_objects/` model code — it takes meshes as
input regardless of how they were produced. Diki's silhouette-guided sampler and other
upstream changes plug in here when they produce a mesh on disk.

## Quick start

```bash
# Single pair, CLI
python -m evaluation.run_eval \
    --predicted  path/to/pred.obj \
    --ground_truth path/to/gt.obj

# Single pair, from Python
from evaluation.run_eval import evaluate_pair
result = evaluate_pair("path/to/pred.obj", "path/to/gt.obj")
# {'chamfer': ..., 'f1@0.005': ..., 'f1@0.01': ..., 'f1@0.02': ..., 'f1@0.05': ...,
#  'n': 10000, 'seed': 0}
```

Both meshes are loaded, **normalized independently** to the `[-1, 1]` cube, then
each gets 10,000 area-weighted surface points before the metrics run. No ICP — feed
in pre-aligned meshes.

## Verifying your install

```bash
python -m evaluation.test_sanity
```

Loads an icosphere, applies controlled perturbations (vertex noise σ, translation),
and asserts the metrics behave correctly:

- **Identity** — same mesh, same seed → Chamfer ≈ 0, F1@0.01 ≈ 1
- **Chamfer monotonicity** — strictly increases as σ ∈ {0, 0.001, 0.005, 0.01, 0.05, 0.1}
- **F-score monotonicity** — F1@0.01 strictly decreases across σ ∈ {0.001, 0.01, 0.05}
- **Symmetry** — Chamfer(A, B) = Chamfer(B, A) to float precision
- **Translation sensitivity** — translating by 0.1 lands Chamfer in (1e-3, 0.5)

Exits 0 on success, raises `AssertionError` on failure.

### Troubleshooting

If you see one of these at startup:

```
Could not find platform independent libraries <prefix>
Fatal Python error: init_fs_encoding: failed to get the Python codec of the filesystem encoding
LookupError: no codec search functions registered: can't find encoding
ModuleNotFoundError: No module named 'encodings'
```

…the conda env on `/scratch-shared/$USER/conda_envs/sam3d-objects/` has had its
Python stdlib source files (`encodings/`, `json/`, `email/`, …) scrubbed by
`/scratch-shared` cleanup — `__pycache__/*.pyc` survives but Python can't bootstrap
from `.pyc` alone. **Recover by rebuilding the env from scratch:**

```bash
rm -rf /scratch-shared/$USER/conda_envs/sam3d-objects
sbatch jobs/01_setup_cpu.job
```

Then re-run `jobs/03_install_inference_gpu.job` if you need the inference extras.

## What's included (MVP)

| File | Purpose |
|------|---------|
| `mesh_io.py` | Load `.obj` / `.ply` / `.glb`; normalize to `[-1, 1]`; save |
| `metrics.py` | `sample_points`, `chamfer`, `f_score`, `voxel_iou`, `emd` |
| `alignment.py` | `align_icp` (committed, not yet wired into `evaluate_pair`) |
| `voxelize.py` | Surface voxelization at 64³ over `[-1, 1]` |
| `perturbations.py` | `translate`, `add_vertex_noise` |
| `run_eval.py` | `evaluate_pair(pred, gt)` + single-pair CLI |
| `test_sanity.py` | Synthetic GT+noise harness |

The active pipeline is **Chamfer + F-score, no ICP, N=10K**. Voxel-IoU, EMD, and ICP
are tested in isolation in `metrics.py` / `alignment.py` but not called from
`evaluate_pair` in this MVP.

## What's not yet included

| Feature | Status | Lands in |
|---------|--------|----------|
| ICP alignment in `evaluate_pair` | Implemented in `alignment.py`, not wired | Phase 6 |
| Voxel-IoU + EMD in `evaluate_pair` | Implemented in `metrics.py`, not wired | Phase 6 |
| `N=1_000_000` paper-protocol mode | Default is `N=10_000` (CPU-friendly) | Phase 6 (GPU-required) |
| `rotate`, `decimate` perturbations | Not implemented | Phase 4 (full synthetic harness) |
| Multi-pair CLI with CSV output | Single-pair CLI only | Phase 6 |
| Dataset loaders (Open3DHOI, SA-3DAO) | `dataset_loader.py` is a stub | Phase 5 |

`synthetic_test.py` (alongside `test_sanity.py`) is a Phase-4 placeholder for the
full harness with rotation, decimation, and soft checks.

## Conventions (pinned)

These choices are documented here so they don't drift. Match the SAM 3D paper §D.3.1
where the MVP overlaps:

1. **Normalization to `[-1, 1]`.** Center bbox at origin, scale longest dim to 2.0.
   Each mesh is normalized independently.
2. **Area-weighted surface sampling**, seeded via `torch.manual_seed` inside a
   `fork_rng` block. Same `(mesh, n, seed)` → same point cloud.
3. **Unsquared symmetric Chamfer.** Mean of (mean unsquared L2 NN distance A→B,
   mean unsquared L2 NN distance B→A). The paper's reported values (full model
   = 0.0400 on SA-3DAO) are in this convention; squared Chamfer would be ~10× smaller
   and not comparable.
4. **F-score thresholds** `τ ∈ {0.005, 0.01, 0.02, 0.05}`. Returned as flat keys
   `f1@0.005`, `f1@0.01`, `f1@0.02`, `f1@0.05` — matches the paper's notation
   (e.g. F1@0.01).
5. **No ICP in MVP.** Inputs must be pre-aligned by the caller. ICP returns in
   Phase 6.

## Ballpark expectations (not paper-comparable)

The SAM 3D paper's full model scores **F1@0.01 = 0.2344**, **Chamfer = 0.0400** on
SA-3DAO. Those numbers come from the *full* protocol (N=1,000,000 + ICP) and are
**not** what this MVP will produce — N=10K + no ICP gives different absolute values.
Treat the paper's numbers as guidance for the *shape* of the answer, not a target
the MVP is trying to hit.

For a well-aligned real prediction at N=10K, expect Chamfer in single-digit hundredths
and F1@0.01 somewhere between ~0.1 and ~0.5. Numbers wildly outside that range
(Chamfer > 0.5, or F1@0.01 = 0 on something that isn't garbage) suggest a bug —
either a misaligned input or a mismatched normalization — not a real quality signal.
Paper-equivalent numbers come back in Phase 6 when ICP and N=1M ship.
