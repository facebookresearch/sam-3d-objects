# Stage 1 Guidance for SAM 3D

This document explains how to run inference with our custom guidance and what it actually does under the hood.

---

## Quick start

```bash
cd ~/sam-3d-objects

python main.py \
  --image  "data/Open3DHOI/data/tennis racket/2344062/image.jpg" \
  --mask   "data/Open3DHOI/data/tennis racket/2344062/obj_mask.png" \
  --prefix "my_run" \
  --seed   42 \
  --ss-guidance-scale    50.0 \
  --pose-guidance-scale   0.5 \
  --depth-guidance-scale 50.0 \
  --depth-map "data/Open3DHOI/data/tennis racket/2344062/depth.npy"
```

Outputs land in `outputs/my_run/`:
- `gaussian.ply` — 3D Gaussian splat (open in [SuperSplat](https://playcanvas.com/supersplat/editor))
- `pred_points.npy` — xyz point cloud as numpy array

---

## All arguments

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Path to RGB image |
| `--mask` | required | Path to object mask (any non-zero = foreground) |
| `--prefix` | required | Output folder name under `outputs/` |
| `--seed` | `42` | RNG seed for reproducibility |
| `--tag` | `hf` | Checkpoint folder under `checkpoints/` |
| `--ss-guidance-scale` | off | Shape guidance strength (try 5–50) |
| `--pose-guidance-scale` | off | Pose guidance strength (try 0.05–1.0) |
| `--depth-guidance-scale` | off | Depth guidance strength (try 5–50); requires `--depth-map` |
| `--depth-map` | off | Path to `depth.npy` from Open3DHOI |
| `--w-centroid` | `1.0` | Weight for centroid term inside pose guidance |
| `--w-size` | `1.0` | Weight for size term inside pose guidance |

You can use any combination of the three guidance types — they compose.

### Baseline (no guidance)
```bash
python main.py --image ... --mask ... --prefix baseline --seed 42
```

### Shape only
```bash
python main.py --image ... --mask ... --prefix shape_only --seed 42 \
  --ss-guidance-scale 10.0
```

### All three
```bash
python main.py --image ... --mask ... --prefix full_guidance --seed 42 \
  --ss-guidance-scale 50.0 --pose-guidance-scale 0.5 \
  --depth-guidance-scale 50.0 --depth-map path/to/depth.npy
```

---

## Running on Snellius (SLURM)

Use the provided job file — it runs baseline and guided back-to-back on the same sample:

```bash
mkdir -p logs
sbatch jobs/05_guidance_test.job
tail -f logs/sam_guidance_test_<JOBID>.out
```

The job file currently requests `gpu_a100`. You can change the `--partition` line to `gpu_h100` if you want to use an H100 instead — both work.

---

## How it works

### Background: Stage 1 of SAM 3D

SAM 3D generates 3D objects in two stages:
1. **Stage 1** samples a *sparse structure* (shape + pose) using a flow-matching ODE solver over ~25 steps
2. **Stage 2** decodes that structure into a 3D Gaussian splat

Our guidance hooks into Stage 1. At each ODE step the solver yields a latent state `x_t` (a dict of tensors for shape, translation, rotation, scale). We compute a loss against a 2D signal from the input image, take a gradient, and nudge `x_t` before the next step.

```
x_t  ──→  decode to mesh  ──→  differentiable render  ──→  loss vs GT mask/depth
  ↑                                                              │
  └──────────────── gradient step (unit-norm) ──────────────────┘
```

This is sometimes called **guidance by latent correction** — similar in spirit to classifier guidance in diffusion models, but applied to a flow-matching ODE.

### ShapeGuidance (`--ss-guidance-scale`)

**Signal:** soft-IoU between the predicted silhouette and the GT object mask.

**What it corrects:** `x_t["shape"]` — the latent that controls the 3D shape.

**How:**
1. Decodes `x_t["shape"]` → voxel grid via the `ss_decoder`
2. Extracts a mesh from the voxel grid using FlexiCubes (differentiable marching cubes)
3. Renders a soft silhouette using PyTorch3D's `SoftSilhouetteShader`
4. Computes `loss = 1 - IoU(rendered, gt_mask)`
5. Backpropagates to get `∂loss/∂x_t["shape"]`, applies a unit-norm gradient step

Higher `--ss-guidance-scale` = stronger pull toward the mask silhouette.

### PoseGuidance (`--pose-guidance-scale`)

**Signal:** centroid position and bounding-box area of the predicted silhouette vs the GT mask.

**What it corrects:** `x_t["translation"]` and `x_t["scale"]`.

**How:**
1. Extracts mesh from shape latent (no grad — mesh is held fixed)
2. Decodes pose latents with grad enabled
3. Renders silhouette, computes:
   - Centroid loss: L2 distance between predicted and GT mask centroid (normalized by image width)
   - Size loss: squared relative error between predicted and GT mask area
4. Separate gradients flow to translation and scale latents

Note: rotation is not corrected — centroid and size give no meaningful rotation signal.

### DepthGuidance (`--depth-guidance-scale`)

**Signal:** scale-invariant depth MSE between the rendered depth map and the GT `depth.npy` from Open3DHOI.

**What it corrects:** `x_t["shape"]`.

**How:**
1. Extracts mesh, renders a hard zbuffer depth map (PyTorch3D rasterizer)
2. Masks to pixels where both predicted and GT depth are valid (non-zero)
3. Normalizes both maps to zero-mean unit-variance before computing MSE (scale-invariant)
4. Backpropagates to shape latent

Requires `--depth-map` pointing to the `depth.npy` file from Open3DHOI.

### CompositeGuidance

All three modules can run together. Each module sees the same `x_t` (corrections don't chain within a step). If two modules correct the same key (e.g. both `ShapeGuidance` and `DepthGuidance` correct `x_t["shape"]`), their gradient steps are summed additively.

---

## Tuning tips

- Start with just one guidance type at a time to understand each signal independently.
- Shape and depth scales (5–50) are typically much larger than pose scale (0.05–1.0) — they operate on different latent spaces with different magnitudes.
- Debug images for `ShapeGuidance` are saved to `outputs/<prefix>/guidance_debug/shape/` (pred / GT / overlap per step) — useful to see if the silhouette is converging.
- If you see `empty mesh — skipping` in the logs for many steps, the shape latent hasn't formed a surface yet; try lowering `--ss-guidance-scale` or using it with `--pose-guidance-scale` to first get the object in the right place.
