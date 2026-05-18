# Stage 1 Guidance for SAM 3D

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

## Arguments

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Path to RGB image |
| `--mask` | required | Path to object mask (any non-zero = foreground) |
| `--prefix` | required | Output folder name under `outputs/` |
| `--seed` | `42` | RNG seed for reproducibility |
| `--tag` | `hf` | Checkpoint folder under `checkpoints/` |
| `--ss-guidance-scale` | off | Shape guidance strength |
| `--pose-guidance-scale` | off | Pose guidance strength |
| `--depth-guidance-scale` | off | Depth guidance strength; requires `--depth-map` |
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

## Running on Snellius

The job file runs baseline and guided back-to-back on the same sample:

```bash
mkdir -p logs
sbatch jobs/05_guidance_test.job
tail -f logs/sam_guidance_test_<JOBID>.out
```

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

All three modules can run together. Each module sees the same `x_t` (corrections don't chain within a step). If two modules correct the same key, their corrections are merged after all modules finish.

---

## Design choices

**Unit-norm gradient step.** The correction applied at each step is:
```python
g = grad / (grad.norm() + 1e-8)
x_t_corrected = x_t - scale * g
```
The gradient is normalized before scaling, so `scale` is a literal step size regardless of gradient magnitude. This keeps the hyperparameter interpretable and stable across ODE timesteps.

**Additive delta composition.** In `CompositeGuidance`, when two modules correct the same key (e.g. both `ShapeGuidance` and `DepthGuidance` write to `x_t["shape"]`), their corrections accumulate as deltas from the original:
```python
corrections[key] = corrections[key] + val - x_t[key]
# → x_t_final = x_t_original + Δ_shape + Δ_depth
```
Each module sees the unmodified `x_t`, so neither gradient computation is contaminated by the other's correction within the same step.

**Partial gradient isolation.** Each module holds fixed the latents it doesn't own: `ShapeGuidance` and `DepthGuidance` decode pose inside `torch.no_grad()`; `PoseGuidance` runs the shape decoder without grad. This prevents cross-contamination — shape gradients don't leak into pose and vice versa.

**Scale-invariant depth loss.** Both predicted and GT depth maps are normalized to zero-mean unit-variance before MSE. The loss only cares about relative depth structure, not absolute scale — important because scene scale varies across samples.

**Rotation excluded from PoseGuidance.** Centroid and size are 2D projective signals with no meaningful 3D rotation information. Correcting rotation from these losses would inject noise, so `x_t["6drotation_normalized"]` is intentionally left untouched.

**`@torch.enable_grad()` inside `torch.no_grad()`.** The ODE loop runs under `torch.no_grad()` for speed. Guidance `apply()` methods are decorated with `@torch.enable_grad()`, which locally re-enables autograd only for the guidance computation. Gradient overhead is zero when guidance is not used.
