import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch3d.transforms import quaternion_to_matrix

from utils import load_config, TrainingLogger
from datasets import BicycleFinetuneDataset

from modules import (
    load_ply_manually,
    save_ply_manually,
    render_standalone,
    initialize_control_points,
    compute_knn_rbf_weights,
    deform_sam_points,
    build_knn_edges_with_threshold,
    remove_duplicate_edges,
    compute_knn_edge_weights,
    print_knn_graph_stats,
    compute_arap_loss,
)
from modules.losses import (
    photo_loss,
    mask_loss,
    compute_total_losses,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_local_TR_photo_arap.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )

    output_dir = cfg.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("========================================")
    print(f"Stage 1: {cfg.experiment.name}")
    print("========================================")
    print(f"device: {device}")
    print(f"output dir: {output_dir}")
    print("========================================")

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

    means = nn.Parameter(m.detach(), requires_grad=False)
    scales = nn.Parameter(s.detach(), requires_grad=False)
    quats = nn.Parameter(q.detach(), requires_grad=False)
    opacities = nn.Parameter(o.detach(), requires_grad=False)
    f_dc = nn.Parameter(f_dc_init.detach(), requires_grad=False)
    f_rest = nn.Parameter(f_rest_init.detach(), requires_grad=False)

    with torch.no_grad():
        static_actual_scales = torch.exp(scales)
        static_norm_quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
        static_actual_opacities = torch.sigmoid(opacities)
        static_colors = 0.5 + 0.28209 * f_dc

    print("initialize control points...")

    c_coord, c_rbf = initialize_control_points(
        means,
        num_points=cfg.control.num_control_points,
    )

    c_coord = c_coord.to(device)
    c_rbf = c_rbf.to(device)

    edge_index, edge_dist = build_knn_edges_with_threshold(
        c_coord,
        k=cfg.control.graph_k,
        threshold=cfg.control.graph_threshold,
    )

    edges = remove_duplicate_edges(edge_index)

    arap_edge_index = edge_index.detach()
    arap_edge_dist = edge_dist.detach()

    arap_edge_weight = compute_knn_edge_weights(
        edge_index=arap_edge_index,
        edge_dist=arap_edge_dist,
        num_points=c_coord.shape[0],
        sigma=None,
    ).detach()

    if cfg.deformer.learn_translation:
        control_T = nn.Parameter(torch.zeros_like(c_coord), requires_grad=True)
    else:
        control_T = nn.Parameter(torch.zeros_like(c_coord), requires_grad=False)

    init_q = torch.zeros(c_coord.shape[0], 4, device=device)
    init_q[:, 0] = 1.0

    if cfg.deformer.learn_rotation:
        control_q = nn.Parameter(init_q, requires_grad=True)
    else:
        control_q = nn.Parameter(init_q, requires_grad=False)

    control_coord = nn.Parameter(
        c_coord.detach(),
        requires_grad=False,
    )

    control_rbf = nn.Parameter(
        c_rbf.detach(),
        requires_grad=False,
    )

    print("compute fixed skinning weights...")

    skinning_weights = compute_knn_rbf_weights(
        means,
        control_coord,
        control_rbf,
    ).detach()

    optim_params = []

    if control_T.requires_grad:
        optim_params.append({
            "params": [control_T],
            "lr": cfg.optimizer.control_point_t_lr,
            "name": "control_T",
        })

    if control_q.requires_grad:
        optim_params.append({
            "params": [control_q],
            "lr": cfg.optimizer.control_point_r_lr,
            "name": "control_q",
        })

    optimizer = torch.optim.Adam(
        optim_params,
        lr=0.0,
        eps=cfg.optimizer.eps,
    )

    total_epochs = max(1, cfg.training.iterations // len(dataset))
    logger = TrainingLogger()

    progress_bar = tqdm(range(1, total_epochs + 1), desc="Epoch Progress")

    for epoch in progress_bar:
        epoch_photo_loss_sum = 0.0
        epoch_arap_loss_sum = 0.0
        epoch_mask_loss_sum = 0.0
        epoch_dt_loss_sum = 0.0
        epoch_total_loss_sum = 0.0

        for _ in range(len(dataset)):
            try:
                camera_dict, gt_image, gt_mask = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                camera_dict, gt_image, gt_mask = next(dataloader_iterator)

            gt_image = gt_image.squeeze(0).to(device, non_blocking=True)
            gt_mask = gt_mask.squeeze(0).to(device, non_blocking=True)
            control_q_norm = torch.nn.functional.normalize(
                control_q,
                p=2,
                dim=-1,
            )

            R_mat_control = quaternion_to_matrix(control_q_norm)

            deformed_means = deform_sam_points(
                means,
                control_coord,
                R_mat_control,
                control_T,
                skinning_weights,
            )

            render_out = render_standalone(
                deformed_means,
                static_actual_scales,
                static_norm_quats,
                static_actual_opacities,
                static_colors,
                camera_dict,
            )

            render_out = render_out.permute(2, 0, 1)
            render_out = render_out.clamp(0.0, 1.0)

            mask_colors = torch.ones_like(static_colors)

            mask_render = render_standalone(
                deformed_means,
                static_actual_scales,
                static_norm_quats,
                static_actual_opacities,
                mask_colors,
                camera_dict,
            )

            mask_render = mask_render.permute(2, 0, 1)
            mask_render = mask_render.clamp(0.0, 1.0)

            pred_mask = mask_render.mean(dim=0, keepdim=True)
            pred_mask = pred_mask.clamp(0.0, 1.0)

            loss_dict = compute_total_losses
            (
                cfg=cfg,
                render_out=render_out,
                gt_image=gt_image,
                gt_mask=gt_mask,
                control_coord=control_coord,
                control_T=control_T,
                control_q=control_q,
                arap_edge_index=arap_edge_index,
                arap_edge_weight=arap_edge_weight,
                pred_mask=pred_mask,
            )

            total_loss = loss_dict["total_loss"]

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            epoch_photo_loss_sum += loss_dict["photo_loss"].item()
            epoch_arap_loss_sum += loss_dict["arap_loss"].item()
            epoch_mask_loss_sum += loss_dict["mask_loss"].item()
            epoch_dt_loss_sum += loss_dict["dt_loss"].item()
            epoch_total_loss_sum += loss_dict["total_loss"].item()

        num_steps = len(dataset)

        epoch_avg_photo_loss = epoch_photo_loss_sum / num_steps
        epoch_avg_arap_loss = epoch_arap_loss_sum / num_steps
        epoch_avg_mask_loss = epoch_mask_loss_sum / num_steps
        epoch_avg_dt_loss = epoch_dt_loss_sum / num_steps
        epoch_avg_total_loss = epoch_total_loss_sum / num_steps

        with torch.no_grad():
            control_T_norm = torch.norm(control_T[:, :3], dim=-1)
            t_mean = control_T_norm.mean().item()
            t_max = control_T_norm.max().item()

        logger.add({
            "Epoch": epoch,
            "Photo_Loss": epoch_avg_photo_loss,
            "ARAP_Loss": epoch_avg_arap_loss,
            "Mask_Loss": epoch_avg_mask_loss,
            "DT_Loss": epoch_avg_dt_loss,
            "Total_Loss": epoch_avg_total_loss,

            "T_Mean": t_mean,
            "T_Max": t_max,
        })

        progress_bar.set_postfix({
            "Total": f"{epoch_avg_total_loss:.5f}",
            "Photo": f"{epoch_avg_photo_loss:.5f}",
            "ARAP": f"{epoch_avg_arap_loss:.5f}",
            "Mask": f"{epoch_avg_mask_loss:.5f}",
            "DT": f"{epoch_avg_dt_loss:.5f}",
            "Tmean": f"{t_mean:.5f}",
            "Tmax": f"{t_max:.5f}",
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
                control_q_norm = torch.nn.functional.normalize(
                    control_q,
                    p=2,
                    dim=-1,
                )
                R_mat_control = quaternion_to_matrix(control_q_norm)

                current_deformed_means = deform_sam_points(
                    means,
                    control_coord,
                    R_mat_control,
                    control_T,
                    skinning_weights,
                )

            save_ply_manually(
                save_path,
                current_deformed_means,
                scales,
                quats,
                opacities,
                f_dc,
                f_rest,
            )

    print("Stage 1 training finished.")


if __name__ == "__main__":
    main()