import argparse
from pathlib import Path

import torch

from utils import load_config

from modules import (
    load_ply_manually,
    save_ply_manually,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_regid_bicycle.yaml",
    )
    return parser.parse_args()


def compute_aabb_from_points(points, margin_ratio=0.25):
    """
    points: torch.Tensor, shape [M, 3]
    """

    bbox_min = points.min(dim=0).values
    bbox_max = points.max(dim=0).values

    bbox_size = bbox_max - bbox_min
    margin = margin_ratio * bbox_size

    crop_min = bbox_min - margin
    crop_max = bbox_max + margin

    return crop_min, crop_max, bbox_min, bbox_max, margin

def crop_scene_by_bbox(scene_xyz, crop_min, crop_max):
    """
    scene_xyz: torch.Tensor, shape [N, 3]
    crop_min: torch.Tensor, shape [3]
    crop_max: torch.Tensor, shape [3]
    """

    inside_min = scene_xyz >= crop_min
    inside_max = scene_xyz <= crop_max

    inside_xyz = inside_min & inside_max

    keep_mask = inside_xyz.all(dim=1)

    crop_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)

    return keep_mask, crop_indices

def resolve_output_path(cfg, path):
    path = Path(path)

    if path.is_absolute():
        return str(path)

    return str(Path(cfg.experiment.output_dir) / path)

def main():
    args = parse_args()
    cfg = load_config(args.config)

    stage2_cfg = cfg.stage2_crop

    aligned_sam3d_ply = resolve_output_path(cfg,stage2_cfg.input.aligned_sam3d_ply)
    scene_ply_path = cfg.model.gt_ply_path

    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )

    margin_ratio = stage2_cfg.crop.margin_ratio

    cropped_gaussians_ply = resolve_output_path(cfg,stage2_cfg.output.cropped_gaussians_ply)
    crop_indices_path = resolve_output_path(cfg,stage2_cfg.output.crop_indices)
    crop_bbox_path = resolve_output_path(cfg,stage2_cfg.output.crop_bbox)

    print("========================================")
    print("Stage 2: Crop Gaussians by Aligned SAM3D")
    print("========================================")
    print(f"Config:           {args.config}")
    print(f"Aligned SAM3D ply:{aligned_sam3d_ply}")
    print(f"Scene 3DGS ply:   {scene_ply_path}")
    print(f"Device:           {device}")
    print(f"Margin ratio:     {margin_ratio}")
    print("----------------------------------------")


    sam3d_xyz, *_ = load_ply_manually(aligned_sam3d_ply)
    sam3d_xyz = sam3d_xyz.to(device)


    scene_means, scene_scales, scene_quats, scene_opacities, scene_f_dc, scene_f_rest = load_ply_manually(
        scene_ply_path
    )

    scene_means = scene_means.to(device)
    scene_scales = scene_scales.to(device)
    scene_quats = scene_quats.to(device)
    scene_opacities = scene_opacities.to(device)
    scene_f_dc = scene_f_dc.to(device)

    if scene_f_rest is not None:
        scene_f_rest = scene_f_rest.to(device)

    print(f"SAM3D points:     {sam3d_xyz.shape[0]}")
    print(f"Scene Gaussians:  {scene_means.shape[0]}")


    crop_min, crop_max, bbox_min, bbox_max, margin = compute_aabb_from_points(
        sam3d_xyz,
        margin_ratio=margin_ratio,
    )

    print("----------------------------------------")
    print("Original SAM3D AABB:")
    print(f"bbox_min: {bbox_min.detach().cpu().numpy()}")
    print(f"bbox_max: {bbox_max.detach().cpu().numpy()}")
    print("Crop AABB with margin:")
    print(f"crop_min: {crop_min.detach().cpu().numpy()}")
    print(f"crop_max: {crop_max.detach().cpu().numpy()}")


    keep_mask, crop_indices = crop_scene_by_bbox(
        scene_means,
        crop_min,
        crop_max,
    )

    num_keep = int(keep_mask.sum().item())
    num_total = int(scene_means.shape[0])
    keep_ratio = num_keep / max(num_total, 1)

    print("----------------------------------------")
    print(f"Kept Gaussians:   {num_keep} / {num_total}")
    print(f"Keep ratio:       {keep_ratio:.6f}")

    if num_keep == 0:
        raise RuntimeError(
            "No Gaussians were kept. Your bbox may be wrong, "
            "or aligned_sam3d.ply and scene_ply_path may be in different coordinate systems."
        )
    cropped_means = scene_means[crop_indices]
    cropped_scales = scene_scales[crop_indices]
    cropped_quats = scene_quats[crop_indices]
    cropped_opacities = scene_opacities[crop_indices]
    cropped_f_dc = scene_f_dc[crop_indices]

    if scene_f_rest is not None:
        cropped_f_rest = scene_f_rest[crop_indices]
    else:
        cropped_f_rest = None

    Path(cropped_gaussians_ply).parent.mkdir(parents=True, exist_ok=True)

    save_ply_manually(
        path=cropped_gaussians_ply,
        means=cropped_means,
        scales=cropped_scales,
        quats=cropped_quats,
        opacities=cropped_opacities,
        f_dc=cropped_f_dc,
        f_rest=cropped_f_rest,
    )


    Path(crop_indices_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(crop_indices.detach().cpu().long(), crop_indices_path)


    Path(crop_bbox_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bbox_min": bbox_min.detach().cpu().float(),
            "bbox_max": bbox_max.detach().cpu().float(),
            "crop_min": crop_min.detach().cpu().float(),
            "crop_max": crop_max.detach().cpu().float(),
            "margin": margin.detach().cpu().float(),
            "margin_ratio": margin_ratio,
            "num_scene_gaussians": num_total,
            "num_cropped_gaussians": num_keep,
            "keep_ratio": keep_ratio,
        },
        crop_bbox_path,
    )

    print("----------------------------------------")
    print("Saved:")
    print(f"Cropped ply:      {cropped_gaussians_ply}")
    print(f"Crop indices:     {crop_indices_path}")
    print(f"Crop bbox:        {crop_bbox_path}")
    print("Done.")


if __name__ == "__main__":
    main()