import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch3d.transforms import quaternion_to_matrix,matrix_to_quaternion
import torch.nn.functional as F
from utils import load_config, TrainingLogger
from datasets import BicycleFinetuneDataset
from pathlib import Path
from modules import (
    load_ply_manually,
    save_ply_manually,
    render_zi,
)
from modules.losses import (
    compute_total_losses_stage3,
)
from modules.losses.stage3.sam3d_loss import (
    build_sam3d_soft_target_from_pointcloud,
)
from modules.losses.stage3.tv_loss import (
    build_knn_graph_for_tv,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_regid_bicycle.yaml",
    )
    return parser.parse_args()
    
def resolve_output_path(cfg, path):
    path = Path(path)

    if path.is_absolute():
        return str(path)

    return str(Path(cfg.experiment.output_dir) / path)

def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(
        cfg.stage3_finetune.training.device if torch.cuda.is_available() else "cpu"
    )

    scene_ply_path = resolve_output_path(cfg, cfg.stage2_crop.output.cropped_gaussians_ply)
    sam3d_ply_path = resolve_output_path(cfg, cfg.stage2_crop.input.aligned_sam3d_ply)

    print("========================================")
    print("Stage 3: Learn Gaussian Membership")
    print("========================================")
    print(f"Cropped scene ply: {scene_ply_path}")
    print(f"Aligned SAM3D ply: {sam3d_ply_path}")
    print(f"Device:            {device}")
    print("----------------------------------------")

    means, scales, quats, opacities, f_dc, f_rest = load_ply_manually(scene_ply_path)

    means = means.to(device).detach()
    scales = scales.to(device).detach()
    quats = quats.to(device).detach()
    opacities = opacities.to(device).detach()
    f_dc = f_dc.to(device).detach()

    # --------------------------------------------------
    # Load aligned SAM3D point cloud
    # --------------------------------------------------
    aligned_sam3d_ply = resolve_output_path(cfg,cfg.stage2_crop.input.aligned_sam3d_ply)

    sam3d_means, *_= load_ply_manually(
        aligned_sam3d_ply
    )

    sam3d_points = sam3d_means.to(device).detach()

    if cfg.stage3_finetune.loss.use_sam3d:
        sam3d_target, sam3d_dist, sam3d_info = build_sam3d_soft_target_from_pointcloud(
            means=means,
            sam3d_points=sam3d_points,
            margin_ratio=cfg.stage3_finetune.loss.sam3d_margin_ratio,
            sigma_ratio=cfg.stage3_finetune.loss.sam3d_sigma_ratio,
        )
    else:
        sam3d_target = None

    # --------------------------------------------------
    # Build TV KNN graph
    # --------------------------------------------------
    if cfg.stage3_finetune.loss.use_tv:
        tv_edge_index, tv_edge_weight, tv_info = build_knn_graph_for_tv(
            means=means,
            k=cfg.stage3_finetune.loss.tv_k,
            use_edge_weight=cfg.stage3_finetune.loss.tv_use_edge_weight,
        )
        print("[TV Graph]")
        print(f"  num points:       {tv_info['num_points']}")
        print(f"  actual k:         {tv_info['actual_k']}")
        print(f"  num edges:        {tv_info['num_edges']}")
        print(f"  edge dist min:    {tv_info['edge_dist_min'].item():.6f}")
        print(f"  edge dist mean:   {tv_info['edge_dist_mean'].item():.6f}")
        print(f"  edge dist median: {tv_info['edge_dist_median'].item():.6f}")
        print(f"  edge dist max:    {tv_info['edge_dist_max'].item():.6f}")
        print(f"  sigma:            {tv_info['sigma'].item():.6f}")
    else:
        tv_edge_index = None
        tv_edge_weight = None

    dataset = BicycleFinetuneDataset(
        data_dir=cfg.data.data_dir,
        image_folder=cfg.data.image_folder,
        mask_folder=cfg.data.mask_folder,
        sparse_folder=cfg.data.sparse_folder,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.training.num_workers > 0 else False,
    )

    dataloader_iterator = iter(dataloader)

    with torch.no_grad():
        static_actual_opacities = torch.sigmoid(opacities)
        static_colors = 0.5 + 0.28209 * f_dc
        quats_norm = F.normalize(quats, p=2, dim=-1)
        static_scales = torch.exp(scales)

    if f_rest is not None:
        f_rest = f_rest.to(device).detach()
    
    num_gaussians = means.shape[0]

    print(f"Number of cropped Gaussians: {num_gaussians}")

    membership_logit = nn.Parameter(torch.zeros(num_gaussians, device=device))

    optimizer = torch.optim.Adam(
        [membership_logit],
        lr=cfg.stage3_finetune.optimizer.zi_lr,
    )

    
    total_epochs = max(1, cfg.stage3_finetune.training.iterations // len(dataset))
    logger = TrainingLogger()

    global_step = 0
    progress_bar = tqdm(range(1, total_epochs + 1), desc="Epoch Progress")
    for epoch in progress_bar:
        epoch_total_loss_sum = 0.0
        epoch_mask_loss_sum = 0.0
        epoch_dt_loss_sum = 0.0
        epoch_sam3d_loss_sum = 0.0
        epoch_tv_loss_sum = 0.0
        
        for _ in range(len(dataset)):
            global_step += 1
            try:
                camera_dict, gt_image, gt_mask = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                camera_dict, gt_image, gt_mask = next(dataloader_iterator)

            gt_image = gt_image.squeeze(0).to(device, non_blocking=True)
            gt_mask = gt_mask.squeeze(0).to(device, non_blocking=True)

            render_mask, membership_prob  = render_zi(
                means=means,
                scales=static_scales,
                quats=quats_norm,
                opacities=static_actual_opacities,
                membership_logit=membership_logit,
                camera_dict=camera_dict,
            )

            loss_dict = compute_total_losses_stage3(
                cfg=cfg,
                render_out=render_mask,
                gt_image=gt_image,
                gt_mask=gt_mask,
                membership_prob=membership_prob,
                sam3d_target=sam3d_target,
                tv_edge_index=tv_edge_index,
                tv_edge_weight=tv_edge_weight,
                step=global_step,
            )

            loss = loss_dict["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_mask_loss_sum += loss_dict["mask_loss"].item()
            epoch_dt_loss_sum += loss_dict["dt_loss"].item()
            epoch_sam3d_loss_sum += loss_dict["sam3d_loss"].item()
            epoch_tv_loss_sum += loss_dict["tv_loss"].item()
            epoch_total_loss_sum += loss_dict["total_loss"].item()


        num_steps = len(dataset)
        epoch_avg_mask_loss = epoch_mask_loss_sum / num_steps
        epoch_avg_dt_loss = epoch_dt_loss_sum / num_steps
        epoch_avg_sam3d_loss = epoch_sam3d_loss_sum / num_steps
        epoch_avg_tv_loss = epoch_tv_loss_sum / num_steps
        epoch_avg_total_loss = epoch_total_loss_sum / num_steps

        logger.add({
            "Epoch": epoch,
            "Mask_Loss": epoch_avg_mask_loss,
            "DT_Loss": epoch_avg_dt_loss,
            "SAM3D_Loss": epoch_avg_sam3d_loss,
            "TV_Loss": epoch_avg_tv_loss,
            "Total_Loss": epoch_avg_total_loss,
        })

        progress_bar.set_postfix({
            "Total": f"{epoch_avg_total_loss:.5f}",
            "Mask": f"{epoch_avg_mask_loss:.5f}",
            "DT": f"{epoch_avg_dt_loss:.5f}",
            "SAM3D": f"{epoch_avg_sam3d_loss:.5f}",
            "TV": f"{epoch_avg_tv_loss:.5f}",
        })

        if epoch % cfg.stage3_finetune.training.save_every == 0 or epoch == total_epochs:
            output_dir=resolve_output_path(cfg, cfg.stage3_finetune.experiment.output_dir)
            csv_path = os.path.join(output_dir, "loss_log.csv")
            logger.save(csv_path)

            save_dir = os.path.join(output_dir, f"finetuned_epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(
                save_dir,
                f"finetuned_epoch_{epoch}.ply",
            )
            with torch.no_grad():
                membership_prob = torch.sigmoid(membership_logit) 

                threshold = 0.5
                selected = membership_prob > threshold

                num_selected = selected.sum().item()
                print(f"[Save] epoch {epoch}, selected Gaussians: {num_selected} / {membership_prob.numel()}")
                quats_norm = F.normalize(quats, p=2, dim=-1)
                if num_selected > 0:
                    save_ply_manually(
                        save_path,
                        means[selected],
                        scales[selected],
                        quats_norm[selected],
                        opacities[selected],
                        f_dc[selected],
                        f_rest[selected] if f_rest is not None else None,
                    )
                else:
                    print(f"[Warning] No Gaussians selected at threshold {threshold}. Skip saving.")

if __name__ == "__main__":
    main()