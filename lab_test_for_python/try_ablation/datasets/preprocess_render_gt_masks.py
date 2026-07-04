import os
import sys
import argparse

import torch
import torchvision.transforms.functional as TF

# ============================================================
# Make project root importable
# File location:
#   try_ablation/datasets/preprocess_render_gt_masks.py
#
# PROJECT_ROOT:
#   try_ablation/
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import load_config
from datasets.bicycle_dataset import BicycleFinetuneDataset
from modules.ply_io import load_ply_manually
from modules.renderer import render_standalone


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render GT wheel masks from gt_ply_path using COLMAP cameras."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_local_TR_mask_arap.yaml",
        help="Path to config file. Relative path will be resolved from project root.",
    )

    parser.add_argument(
        "--output-mask-folder",
        type=str,
        default="wheel_masks_rendered",
        help="Folder name under cfg.data.data_dir to save rendered GT wheel masks.",
    )

    parser.add_argument(
        "--input-mask-folder",
        type=str,
        default="masks",
        help=(
            "Existing mask folder used only to satisfy BicycleFinetuneDataset. "
            "This preprocessing script does not use these masks as GT target."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Threshold for converting rendered soft mask to binary mask.",
    )

    return parser.parse_args()


def resolve_path(path):
    """
    If path is relative, resolve it from PROJECT_ROOT.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def main():
    args = parse_args()

    config_path = resolve_path(args.config)
    cfg = load_config(config_path)

    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    data_dir = cfg.data.data_dir

    output_mask_dir = os.path.join(
        data_dir,
        args.output_mask_folder,
    )

    os.makedirs(output_mask_dir, exist_ok=True)

    gt_ply_path = cfg.model.gt_ply_path

    print("========================================")
    print("Render GT wheel masks")
    print("========================================")
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Config path        : {config_path}")
    print(f"Data dir           : {data_dir}")
    print(f"GT PLY             : {gt_ply_path}")
    print(f"Input mask folder  : {args.input_mask_folder}")
    print(f"Output mask folder : {output_mask_dir}")
    print(f"Threshold          : {args.threshold}")
    print(f"Device             : {device}")
    print("========================================")

    # ------------------------------------------------------------
    # Dataset
    # Important:
    #   這裡 input-mask-folder 只是為了讓 BicycleFinetuneDataset 可以正常 __getitem__。
    #   這支 script 真正輸出的 mask 來自 gt_ply_path render。
    # ------------------------------------------------------------
    dataset = BicycleFinetuneDataset(
        data_dir=data_dir,
        image_folder=cfg.data.image_folder,
        mask_folder=args.input_mask_folder,
        sparse_folder=cfg.data.sparse_folder,
    )

    # ------------------------------------------------------------
    # Load GT wheel PLY
    # ------------------------------------------------------------
    means, scales, quats, opacities, f_dc, f_rest = load_ply_manually(
        gt_ply_path,
    )

    means = means.to(device)
    scales = scales.to(device)
    quats = quats.to(device)
    opacities = opacities.to(device)
    f_dc = f_dc.to(device)

    with torch.no_grad():
        actual_scales = torch.exp(scales)
        norm_quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
        actual_opacities = torch.sigmoid(opacities)

        # Render GT object as white, then convert RGB render to mask
        white_colors = torch.ones_like(f_dc)

    # ------------------------------------------------------------
    # Render GT mask for each camera
    # ------------------------------------------------------------
    for idx in range(len(dataset)):
        camera_dict, gt_image, _ = dataset[idx]

        with torch.no_grad():
            mask_render = render_standalone(
                means,
                actual_scales,
                norm_quats,
                actual_opacities,
                white_colors,
                camera_dict,
            )

            # [H, W, 3] -> [3, H, W]
            mask_render = mask_render.permute(2, 0, 1).clamp(0.0, 1.0)

            # [3, H, W] -> [1, H, W]
            soft_mask = mask_render.mean(dim=0, keepdim=True).clamp(0.0, 1.0)

            # Binary GT wheel mask
            binary_mask = (soft_mask > args.threshold).float()

        image_name = camera_dict["image_name"]
        base_name = os.path.splitext(image_name)[0]
        mask_name = base_name + "_mask.png"

        save_path = os.path.join(output_mask_dir, mask_name)

        TF.to_pil_image(binary_mask.cpu()).save(save_path)

        if idx % 20 == 0:
            print(f"[{idx:04d}/{len(dataset)}] saved: {save_path}")

    print("========================================")
    print("Finished rendering GT masks.")
    print(f"Saved to: {output_mask_dir}")
    print("========================================")


if __name__ == "__main__":
    main()