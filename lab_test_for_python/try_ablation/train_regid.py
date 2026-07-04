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

from modules import (
    load_ply_manually,
    save_ply_manually,
    render_standalone,
)
from modules.losses import (
    compute_total_losses,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_regid.yaml",
    )
    return parser.parse_args()

def apply_global_transform(means,scales,quats,global_rotation,global_translation,raw_scale,):
    global_rotation_norm = F.normalize(global_rotation, p=2, dim=0)
    R_global = quaternion_to_matrix(global_rotation_norm)

    quats_norm = F.normalize(quats, p=2, dim=-1)
    R_local = quaternion_to_matrix(quats_norm)

    global_scale = torch.exp(raw_scale)

    aligned_means = global_scale * (means @ R_global.T) + global_translation

    aligned_raw_scales = scales + raw_scale

    aligned_actual_scales = torch.exp(aligned_raw_scales)

    R_mat = R_global @ R_local
    q_aligned = matrix_to_quaternion(R_mat)
    q_aligned_norm = F.normalize(q_aligned, p=2, dim=-1)

    return aligned_means, aligned_actual_scales, aligned_raw_scales, q_aligned_norm

def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )

    output_dir = cfg.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("="*50)
    print("train only regid (parameter : rotation ,traslation and scale)")
    print("="*50)

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
    print("load SAM3D PLY...")
    m, s, q, o, f_dc_init, f_rest_init = load_ply_manually(
        cfg.model.sam3d_ply_path,
    )
    
    means = m.detach().to(device)
    scales = s.detach().to(device)
    quats = q.detach().to(device)
    opacities = o.detach().to(device)
    f_dc = f_dc_init.detach().to(device)
    f_rest = f_rest_init.detach().to(device)

    global_translation = nn.Parameter(torch.zeros(3, device=device), requires_grad=True)
    global_rotation = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device), requires_grad=True)
    raw_scale = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)

    with torch.no_grad():
        static_actual_opacities = torch.sigmoid(opacities)
        static_colors = 0.5 + 0.28209 * f_dc
    optimizer = torch.optim.Adam([
        {
            "params": [global_translation],
            "lr": cfg.optimizer.lr_translation,
        },
        {
            "params": [global_rotation],
            "lr": cfg.optimizer.lr_rotation,
        },
        {
            "params": [raw_scale],
            "lr": cfg.optimizer.lr_scale,
        },
    ],
    eps=cfg.optimizer.eps,
    )

    total_epochs = max(1, cfg.training.iterations // len(dataset))
    logger = TrainingLogger()

    progress_bar = tqdm(range(1, total_epochs + 1), desc="Epoch Progress")
    for epoch in progress_bar:
            epoch_dt_loss_sum = 0.0
            epoch_mask_loss_sum = 0.0
            epoch_total_loss_sum = 0.0
            epoch_mask_ratio = 0.0
            epoch_dt_ratio = 0.0

            for _ in range(len(dataset)):
                try:
                    camera_dict, gt_image, gt_mask = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(dataloader)
                    camera_dict, gt_image, gt_mask = next(dataloader_iterator)

                gt_image = gt_image.squeeze(0).to(device, non_blocking=True)
                gt_mask = gt_mask.squeeze(0).to(device, non_blocking=True)

                aligned_means, aligned_actual_scales, aligned_raw_scales, q_aligned_norm = apply_global_transform(
                    means,
                    scales,
                    quats,
                    global_rotation,
                    global_translation,
                    raw_scale,
                )

                render_out = render_standalone(
                    aligned_means,
                    aligned_actual_scales,
                    q_aligned_norm,
                    static_actual_opacities,
                    static_colors,
                    camera_dict,
                )

                render_out = render_out.permute(2, 0, 1)
                render_out = render_out.clamp(0.0, 1.0)

                mask_colors = torch.ones_like(static_colors)

                mask_render = render_standalone(
                    aligned_means,
                    aligned_actual_scales,
                    q_aligned_norm,
                    static_actual_opacities,
                    mask_colors,
                    camera_dict,
                )

                mask_render = mask_render.permute(2, 0, 1)
                mask_render = mask_render.clamp(0.0, 1.0)

                pred_mask = mask_render.mean(dim=0, keepdim=True)
                pred_mask = pred_mask.clamp(0.0, 1.0)

                loss_dict = compute_total_losses(
                    cfg=cfg,
                    render_out=render_out,
                    gt_image=gt_image,
                    gt_mask=gt_mask,
                    control_coord=None,
                    control_T=None,
                    control_q=None,
                    arap_edge_index=None,
                    arap_edge_weight=None,
                    pred_mask=pred_mask,
                )

                total_loss = loss_dict["total_loss"]

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

                epoch_mask_loss_sum += loss_dict["mask_loss"].item()
                epoch_dt_loss_sum += loss_dict["dt_loss"].item()
                epoch_total_loss_sum += loss_dict["total_loss"].item()
                epoch_mask_ratio += loss_dict["mask_ratio"].item()
                epoch_dt_ratio += loss_dict["dt_ratio"].item()
            num_steps = len(dataset)
            epoch_avg_mask_loss = epoch_mask_loss_sum / num_steps
            epoch_avg_dt_loss = epoch_dt_loss_sum / num_steps
            epoch_avg_total_loss = epoch_total_loss_sum / num_steps
            epoch_mask_ratio = epoch_mask_ratio / num_steps
            epoch_dt_ratio = epoch_dt_ratio / num_steps

            logger.add({
                "Epoch": epoch,
                "Mask_Loss": epoch_avg_mask_loss,
                "DT_Loss": epoch_avg_dt_loss,
                "Total_Loss": epoch_avg_total_loss,
                "Mask_Ratio": epoch_mask_ratio,
                "DT_Ratio": epoch_dt_ratio,
            })

            progress_bar.set_postfix({
                "Total": f"{epoch_avg_total_loss:.5f}",
                "Mask": f"{epoch_avg_mask_loss:.5f}",
                "DT": f"{epoch_avg_dt_loss:.5f}",
                "Mask_Ratio": f"{epoch_mask_ratio:.5f}",
                "DT_Ratio": f"{epoch_dt_ratio:.5f}",
            })

            if epoch % cfg.training.save_every == 0 or epoch == total_epochs:
                csv_path = os.path.join(output_dir, "loss_log.csv")
                logger.save(csv_path)

                save_dir = os.path.join(output_dir, f"finetuned_epoch_{epoch}")
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(
                    save_dir,
                    f"finetuned_epoch_{epoch}.ply",
                )
                with torch.no_grad():
                    aligned_means, aligned_actual_scales, aligned_raw_scales, q_aligned_norm = apply_global_transform(
                        means,
                        scales,
                        quats,
                        global_rotation,
                        global_translation,
                        raw_scale,
                    )
                
                    save_ply_manually(
                        save_path,
                        aligned_means,
                        aligned_raw_scales,
                        q_aligned_norm,
                        opacities,
                        f_dc,
                        f_rest,
                    )


    print("global alignment finetuning completed.")

if __name__ == "__main__":
    main()