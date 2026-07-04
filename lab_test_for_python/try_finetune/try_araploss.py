import os
import warnings
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
os.environ["LIDRA_SKIP_INIT"] = "true"
warnings.filterwarnings("ignore", category=FutureWarning)

import math
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch3d.transforms import quaternion_to_matrix

from data_preprocessing import BicycleFinetuneDataset

from util import (
    render_standalone,
    load_ply_manually,
    save_ply_manually,
    get_ssim_loss,
    get_l1_loss,
    initialize_control_points,
    compute_knn_rbf_weights,
    deform_sam_points
)


# ============================================================
# KNN graph construction
# ============================================================

def build_knn_edges_with_threshold(c_coord, k=8, threshold=None):
    """
    Build directed KNN edges for control points.

    Args:
        c_coord: [N, D], first 3 dims are xyz.
        k: number of neighbors.
        threshold: distance threshold. If not None, remove edges longer than threshold.

    Returns:
        edge_index: [2, E], directed edges i -> k.
        edge_dist: [E], edge distances.
    """
    device = c_coord.device
    xyz = c_coord[:, :3]
    N = xyz.shape[0]

    dist = torch.cdist(xyz, xyz)
    dist.fill_diagonal_(float("inf"))

    k = min(k, N - 1)

    knn_dist, knn_idx = torch.topk(
        dist,
        k=k,
        dim=1,
        largest=False
    )

    src = torch.arange(N, device=device).view(N, 1).expand(N, k)
    dst = knn_idx

    if threshold is not None:
        mask = knn_dist < threshold
        src = src[mask]
        dst = dst[mask]
        edge_dist = knn_dist[mask]
    else:
        src = src.reshape(-1)
        dst = dst.reshape(-1)
        edge_dist = knn_dist.reshape(-1)

    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, edge_dist


def remove_duplicate_edges(edge_index):
    """
    Convert directed edges into unique undirected edges.
    Mainly used for visualization.
    """
    src = edge_index[0]
    dst = edge_index[1]

    edge_min = torch.minimum(src, dst)
    edge_max = torch.maximum(src, dst)

    edges = torch.stack([edge_min, edge_max], dim=1)
    unique_edges = torch.unique(edges, dim=0)

    return unique_edges


def print_knn_graph_stats(c_coord, edge_index, edges):
    """
    Print graph statistics for debugging.
    """
    xyz = c_coord[:, :3]
    num_points = xyz.shape[0]

    bbox_min = xyz.min(dim=0).values
    bbox_max = xyz.max(dim=0).values
    bbox_size = bbox_max - bbox_min
    bbox_diag = torch.norm(bbox_size)

    print("========== KNN Graph Stats ==========")
    print(f"num control points: {num_points}")
    print(f"directed edges: {edge_index.shape[1]}")
    print(f"unique undirected edges: {edges.shape[0]}")
    print(f"bbox min: {bbox_min.detach().cpu().numpy()}")
    print(f"bbox max: {bbox_max.detach().cpu().numpy()}")
    print(f"bbox size: {bbox_size.detach().cpu().numpy()}")
    print(f"bbox diagonal: {bbox_diag.item():.6f}")

    if edges.shape[0] > 0:
        i = edges[:, 0]
        j = edges[:, 1]
        edge_len = torch.norm(xyz[i] - xyz[j], dim=-1)

        print(f"edge length min: {edge_len.min().item():.6f}")
        print(f"edge length max: {edge_len.max().item():.6f}")
        print(f"edge length mean: {edge_len.mean().item():.6f}")

        degree = torch.zeros(num_points, device=xyz.device)
        degree.scatter_add_(0, edges[:, 0], torch.ones(edges.shape[0], device=xyz.device))
        degree.scatter_add_(0, edges[:, 1], torch.ones(edges.shape[0], device=xyz.device))

        print(f"degree min: {degree.min().item():.0f}")
        print(f"degree max: {degree.max().item():.0f}")
        print(f"degree mean: {degree.mean().item():.2f}")
        print(f"num isolated points: {(degree == 0).sum().item()}")

    print("=====================================")


def save_as_ply_with_bright_edges(filename, c_coord, edges):
    """
    Save control points and KNN edges as PLY for MeshLab visualization.
    """
    xyz = c_coord[:, :3].detach().cpu().numpy()
    edges_np = edges.detach().cpu().numpy()

    num_vertices = xyz.shape[0]
    num_edges = edges_np.shape[0]

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")

        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write(f"element edge {num_edges}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write("end_header\n")

        for p in xyz:
            x, y, z = p
            f.write(f"{x} {y} {z} 255 40 40\n")

        for e in edges_np:
            i, j = int(e[0]), int(e[1])
            f.write(f"{i} {j} 40 255 40\n")

    print(f"Saved KNN graph PLY to: {filename}")
    print(f"Vertices: {num_vertices}, Edges: {num_edges}")


# ============================================================
# ARAP edge weights
# ============================================================

def compute_knn_edge_weights(edge_index, edge_dist, num_points, sigma=None, eps=1e-8):
    """
    Compute normalized RBF weights for directed control-control edges.

    For each source point i:
        sum_k w_ik ≈ 1

    Args:
        edge_index: [2, E]
        edge_dist: [E]
        num_points: number of control points
        sigma: RBF sigma. If None, use mean edge distance.

    Returns:
        edge_weight: [E]
    """
    device = edge_dist.device
    src = edge_index[0]

    if sigma is None:
        sigma = edge_dist.mean().detach()

    raw_weight = torch.exp(-(edge_dist ** 2) / (2.0 * sigma ** 2 + eps))

    weight_sum = torch.zeros(num_points, device=device)
    weight_sum.scatter_add_(0, src, raw_weight)

    edge_weight = raw_weight / (weight_sum[src] + eps)

    return edge_weight


# ============================================================
# ARAP losses
# ============================================================

def weighted_arap_edge_length_loss(
    control_coord,
    control_T,
    edge_index,
    edge_weight,
    normalized=True,
    eps=1e-8
):
    """
    Weighted edge-length ARAP.

    Objective:
        ||c_i' - c_k'|| ≈ ||c_i - c_k||

    If normalized=True:
        loss is normalized by rest edge length squared.
        This makes ARAP scale easier to tune.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=control_coord.device)

    xyz_rest = control_coord[:, :3]
    xyz_deformed = control_coord[:, :3] + control_T[:, :3]

    i = edge_index[0]
    k = edge_index[1]

    rest_edge = xyz_rest[i] - xyz_rest[k]
    deformed_edge = xyz_deformed[i] - xyz_deformed[k]

    rest_len = torch.norm(rest_edge, dim=-1)
    deformed_len = torch.norm(deformed_edge, dim=-1)

    loss_per_edge = (deformed_len - rest_len) ** 2

    if normalized:
        rest_len_sq = (rest_len ** 2).clamp_min(eps)
        loss_per_edge = loss_per_edge / rest_len_sq

    loss = torch.sum(edge_weight * loss_per_edge) / (torch.sum(edge_weight) + eps)

    return loss


def weighted_arap_rotation_loss(
    control_coord,
    control_T,
    control_q,
    edge_index,
    edge_weight,
    normalized=True,
    eps=1e-8
):
    """
    Weighted rotation-based ARAP.

    Objective:
        (c_i' - c_k') ≈ R_i (c_i - c_k)

    where:
        c_i' = c_i + T_i

    If normalized=True:
        loss is normalized by rest edge length squared.
        This makes the loss represent relative local distortion.
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=control_coord.device)

    xyz_rest = control_coord[:, :3]
    xyz_deformed = control_coord[:, :3] + control_T[:, :3]

    q_norm = torch.nn.functional.normalize(control_q, p=2, dim=-1)
    R = quaternion_to_matrix(q_norm)

    i = edge_index[0]
    k = edge_index[1]

    rest_edge = xyz_rest[i] - xyz_rest[k]
    deformed_edge = xyz_deformed[i] - xyz_deformed[k]

    rotated_rest_edge = torch.bmm(
        R[i],
        rest_edge.unsqueeze(-1)
    ).squeeze(-1)

    loss_per_edge = torch.sum((deformed_edge - rotated_rest_edge) ** 2, dim=-1)

    if normalized:
        rest_len_sq = torch.sum(rest_edge ** 2, dim=-1).clamp_min(eps)
        loss_per_edge = loss_per_edge / rest_len_sq

    loss = torch.sum(edge_weight * loss_per_edge) / (torch.sum(edge_weight) + eps)

    return loss


def compute_arap_loss(
    control_coord,
    control_T,
    control_q,
    edge_index,
    edge_weight,
    arap_type,
    normalized=True
):
    """
    Select ARAP loss type.
    """
    if arap_type == "length":
        return weighted_arap_edge_length_loss(
            control_coord=control_coord,
            control_T=control_T,
            edge_index=edge_index,
            edge_weight=edge_weight,
            normalized=normalized
        )

    elif arap_type == "rotation":
        return weighted_arap_rotation_loss(
            control_coord=control_coord,
            control_T=control_T,
            control_q=control_q,
            edge_index=edge_index,
            edge_weight=edge_weight,
            normalized=normalized
        )

    elif arap_type == "both":
        loss_len = weighted_arap_edge_length_loss(
            control_coord=control_coord,
            control_T=control_T,
            edge_index=edge_index,
            edge_weight=edge_weight,
            normalized=normalized
        )

        loss_rot = weighted_arap_rotation_loss(
            control_coord=control_coord,
            control_T=control_T,
            control_q=control_q,
            edge_index=edge_index,
            edge_weight=edge_weight,
            normalized=normalized
        )

        return loss_rot + 0.1 * loss_len

    else:
        raise ValueError(f"Unknown arap_type: {arap_type}")


# ============================================================
# Plot helper
# ============================================================

def plot_simple_training_results(csv_path, output_dir):
    """
    Simple plotting function for current CSV format.
    """
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["Epoch"], df["Photo_Loss"], label="Photo Loss")
    plt.plot(df["Epoch"], df["Total_Loss"], label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_photo_total.png"))
    plt.close()

    plt.figure()
    plt.plot(df["Epoch"], df["Raw_ARAP_Loss"], label="Raw ARAP Loss")
    plt.plot(df["Epoch"], df["Weighted_ARAP_Loss"], label="Weighted ARAP Loss")
    plt.xlabel("Epoch")
    plt.ylabel("ARAP Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_arap.png"))
    plt.close()

    plt.figure()
    plt.plot(df["Epoch"], df["ARAP_to_Photo_Ratio"], label="ARAP / Photo")
    plt.xlabel("Epoch")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "arap_photo_ratio.png"))
    plt.close()

    plt.figure()
    plt.plot(df["Epoch"], df["T_Mean"], label="Control T Mean")
    plt.plot(df["Epoch"], df["T_Max"], label="Control T Max")
    plt.xlabel("Epoch")
    plt.ylabel("Translation Norm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "control_translation_norm.png"))
    plt.close()


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="3DGS Bicycle Fine-tuning with Weighted ARAP Loss")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/work/goet1019/dataset/mip_nerf360/3dgs_camera_pose/bicycle"
    )
    parser.add_argument(
        "--initial_sam3d_ply_path",
        type=str,
        default="/work/goet1019/sam-3d-objects/lab_test_for_python/some_basic_pointcloud/onlywheel_sam3d.ply",
        help="Path to SAM3D PLY"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/wheel_finetuned"
    )

    parser.add_argument("--iterations", type=int, default=19400)

    # 3DGS attribute learning rates, currently not used in stage 1
    parser.add_argument("--position_lr", type=float, default=0.0016)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)

    # Photo loss
    parser.add_argument("--lambda_dssim", type=float, default=0.2)

    # Control point learning rates
    parser.add_argument("--control_point_r_lr", type=float, default=0.0005)
    parser.add_argument("--control_point_t_lr", type=float, default=0.001)
    parser.add_argument("--control_point_coord_lr", type=float, default=0.0001)
    parser.add_argument("--control_point_rbf_lr", type=float, default=0.0001)

    # Control point settings
    parser.add_argument("--num_control_points", type=int, default=1024)

    # KNN graph settings
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--knn_threshold", type=float, default=0.07)

    # ARAP settings
    parser.add_argument("--arap_type", type=str, default="rotation", choices=["length", "rotation", "both"])
    parser.add_argument("--lambda_arap", type=float, default=2.5)
    parser.add_argument("--normalized_arap", action="store_true", default=True)
    parser.add_argument("--no_normalized_arap", dest="normalized_arap", action="store_false")

    # Debug / saving
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_knn_ply", action="store_true")
    parser.add_argument("--disable_ssim", action="store_true")

    return parser.parse_args()


# ============================================================
# Main training
# ============================================================

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path_target = os.path.join(
        args.output_dir,
        f"stage1_{args.arap_type}"
        f"_lam{args.lambda_arap}"
        f"_k{args.knn_k}"
        f"_th{args.knn_threshold}"
        f"_cp{args.num_control_points}"
        f"_norm{int(args.normalized_arap)}"
    )
    os.makedirs(save_path_target, exist_ok=True)

    print("========================================")
    print("Experiment setting")
    print("========================================")
    print(f"device: {device}")
    print(f"output dir: {save_path_target}")
    print(f"arap_type: {args.arap_type}")
    print(f"lambda_arap: {args.lambda_arap}")
    print(f"normalized_arap: {args.normalized_arap}")
    print(f"knn_k: {args.knn_k}")
    print(f"knn_threshold: {args.knn_threshold}")
    print(f"num_control_points: {args.num_control_points}")
    print(f"lambda_dssim: {args.lambda_dssim}")
    print(f"disable_ssim: {args.disable_ssim}")
    print("========================================")

    # ----------------------------
    # Dataset
    # ----------------------------
    dataset = BicycleFinetuneDataset(data_dir=args.data_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    dataloader_iterator = iter(dataloader)

    # ----------------------------
    # Logs
    # ----------------------------
    total_photo_loss_log = []
    total_raw_arap_loss_log = []
    total_weighted_arap_loss_log = []
    total_loss_log = []
    arap_ratio_log = []
    lambda_arap_log = []
    t_mean_log = []
    t_max_log = []

    # ----------------------------
    # Load SAM3D / 3DGS PLY
    # ----------------------------
    print("load ply data...")

    m, s, q, o, f_dc_init, f_rest_init = load_ply_manually(args.initial_sam3d_ply_path)

    m = m.to(device)
    s = s.to(device)
    q = q.to(device)
    o = o.to(device)
    f_dc_init = f_dc_init.to(device)
    f_rest_init = f_rest_init.to(device)

    # ----------------------------
    # Initialize control points
    # ----------------------------
    c_coord, c_rbf = initialize_control_points(m, num_points=args.num_control_points)
    c_coord = c_coord.to(device)
    c_rbf = c_rbf.to(device)

    # ----------------------------
    # Build KNN graph on control points
    # ----------------------------
    edge_index, edge_dist = build_knn_edges_with_threshold(
        c_coord,
        k=args.knn_k,
        threshold=args.knn_threshold
    )

    edges = remove_duplicate_edges(edge_index)
    print_knn_graph_stats(c_coord, edge_index, edges)

    arap_edge_index = edge_index.detach()
    arap_edge_dist = edge_dist.detach()

    arap_edge_weight = compute_knn_edge_weights(
        edge_index=arap_edge_index,
        edge_dist=arap_edge_dist,
        num_points=c_coord.shape[0],
        sigma=None
    ).detach()

    if args.save_knn_ply:
        knn_ply_path = os.path.join(save_path_target, "control_points_knn_graph.ply")
        save_as_ply_with_bright_edges(knn_ply_path, c_coord, edges)

    # ----------------------------
    # Stage 1 trainable parameters
    # ----------------------------
    control_T = nn.Parameter(torch.zeros_like(c_coord), requires_grad=True)

    init_q = torch.zeros(c_coord.shape[0], 4, device=device)
    init_q[:, 0] = 1.0
    control_q = nn.Parameter(init_q, requires_grad=True)

    # Frozen control point attributes
    control_coord = nn.Parameter(c_coord.detach(), requires_grad=False)
    control_rbf = nn.Parameter(c_rbf.detach(), requires_grad=False)

    # Frozen Gaussian attributes
    means = nn.Parameter(m.detach(), requires_grad=False)
    scales = nn.Parameter(s.detach(), requires_grad=False)
    quats = nn.Parameter(q.detach(), requires_grad=False)
    opacities = nn.Parameter(o.detach(), requires_grad=False)
    f_dc = nn.Parameter(f_dc_init.detach(), requires_grad=False)
    f_rest = nn.Parameter(f_rest_init.detach(), requires_grad=False)

    # ----------------------------
    # Static Gaussian values
    # ----------------------------
    with torch.no_grad():
        initial_physical_scales = torch.exp(scales)
        base_scale = torch.quantile(initial_physical_scales.view(-1), 0.95).item()
        max_raw_scale = math.log(base_scale * 5.0)

        print(f"the maximum scale in training is set to {max_raw_scale:.4f} (base scale: {base_scale:.4f})")

        static_actual_scales = torch.exp(scales)
        static_norm_quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
        static_actual_opacities = torch.sigmoid(opacities)
        static_colors = 0.5 + 0.28209 * f_dc

    # ----------------------------
    # Compute skinning weights once
    # ----------------------------
    print("計算初始 RBF 權重...")
    skinning_weights = compute_knn_rbf_weights(
        means,
        control_coord,
        control_rbf
    ).detach()

    # ----------------------------
    # Optimizer
    # ----------------------------
    optimizer = torch.optim.Adam(
        [
            {"params": [control_T], "lr": args.control_point_t_lr, "name": "control_T"},
            {"params": [control_q], "lr": args.control_point_r_lr, "name": "control_q"},
        ],
        lr=0.0,
        eps=1e-15
    )

    # ----------------------------
    # Epoch setting
    # ----------------------------
    total_epochs = max(1, args.iterations // len(dataset))
    stage_1_epochs = total_epochs // 2

    print(f"total {total_epochs} epochs...")
    print(f"stage 1 epoch: {stage_1_epochs}")
    print("stage 1: training control point translation and rotation...")

    progress_bar = tqdm(range(1, stage_1_epochs + 1), desc="Epoch Progress")

    global_step = 0

    # ========================================================
    # Training loop
    # ========================================================
    for epoch in progress_bar:
        epoch_photo_loss_sum = 0.0
        epoch_raw_arap_loss_sum = 0.0
        epoch_weighted_arap_loss_sum = 0.0
        epoch_total_loss_sum = 0.0

        for each_step in range(len(dataset)):
            try:
                camera_dict, gt_image, gt_mask = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                camera_dict, gt_image, gt_mask = next(dataloader_iterator)

            gt_image = gt_image.squeeze(0).to(device, non_blocking=True)
            gt_mask = gt_mask.squeeze(0).to(device, non_blocking=True)

            masked_gt = gt_image * gt_mask

            # Each iteration recomputes control point rotation.
            control_q_norm = torch.nn.functional.normalize(control_q, p=2, dim=-1)
            R_mat_control = quaternion_to_matrix(control_q_norm)

            deformed_means = deform_sam_points(
                means,
                control_coord,
                R_mat_control,
                control_T,
                skinning_weights
            )

            render_out = render_standalone(
                deformed_means,
                static_actual_scales,
                static_norm_quats,
                static_actual_opacities,
                static_colors,
                camera_dict
            )

            render_out = render_out.permute(2, 0, 1)
            render_out = render_out.clamp(0.0, 1.0)

            masked_rendered = render_out * gt_mask

            # ----------------------------
            # Photo loss
            # ----------------------------
            loss_L1 = get_l1_loss(masked_rendered, masked_gt)

            if args.disable_ssim or args.lambda_dssim <= 0:
                loss_photo = loss_L1
            else:
                loss_ssim = get_ssim_loss(masked_rendered, masked_gt)
                loss_photo = (1.0 - args.lambda_dssim) * loss_L1 + args.lambda_dssim * loss_ssim

            # ----------------------------
            # ARAP loss
            # ----------------------------
            loss_arap = compute_arap_loss(
                control_coord=control_coord,
                control_T=control_T,
                control_q=control_q,
                edge_index=arap_edge_index,
                edge_weight=arap_edge_weight,
                arap_type=args.arap_type,
                normalized=args.normalized_arap
            )

            loss_arap_weighted = args.lambda_arap * loss_arap

            # ----------------------------
            # Total loss
            # ----------------------------
            total_loss = loss_photo + loss_arap_weighted

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            global_step += 1

            epoch_photo_loss_sum += loss_photo.item()
            epoch_raw_arap_loss_sum += loss_arap.item()
            epoch_weighted_arap_loss_sum += loss_arap_weighted.item()
            epoch_total_loss_sum += total_loss.item()

        # ----------------------------
        # Epoch average
        # ----------------------------
        num_steps = len(dataset)

        epoch_avg_photo_loss = epoch_photo_loss_sum / num_steps
        epoch_avg_raw_arap_loss = epoch_raw_arap_loss_sum / num_steps
        epoch_avg_weighted_arap_loss = epoch_weighted_arap_loss_sum / num_steps
        epoch_avg_total_loss = epoch_total_loss_sum / num_steps
        arap_ratio = epoch_avg_weighted_arap_loss / (epoch_avg_photo_loss + 1e-8)

        with torch.no_grad():
            control_T_norm = torch.norm(control_T[:, :3], dim=-1)
            t_mean = control_T_norm.mean().item()
            t_max = control_T_norm.max().item()

        total_photo_loss_log.append(epoch_avg_photo_loss)
        total_raw_arap_loss_log.append(epoch_avg_raw_arap_loss)
        total_weighted_arap_loss_log.append(epoch_avg_weighted_arap_loss)
        total_loss_log.append(epoch_avg_total_loss)
        arap_ratio_log.append(arap_ratio)
        lambda_arap_log.append(args.lambda_arap)
        t_mean_log.append(t_mean)
        t_max_log.append(t_max)

        progress_bar.set_postfix({
            "Total": f"{epoch_avg_total_loss:.5f}",
            "Photo": f"{epoch_avg_photo_loss:.5f}",
            "RawA": f"{epoch_avg_raw_arap_loss:.5e}",
            "wA": f"{epoch_avg_weighted_arap_loss:.5f}",
            "Ratio": f"{arap_ratio:.3f}",
            "Tmean": f"{t_mean:.5f}",
            "Tmax": f"{t_max:.5f}",
        })

        # ----------------------------
        # Save log and checkpoint
        # ----------------------------
        if epoch % args.save_every == 0 or epoch == stage_1_epochs:
            df = pd.DataFrame({
                "Epoch": range(1, len(total_loss_log) + 1),
                "Photo_Loss": total_photo_loss_log,
                "Raw_ARAP_Loss": total_raw_arap_loss_log,
                "Weighted_ARAP_Loss": total_weighted_arap_loss_log,
                "Total_Loss": total_loss_log,
                "ARAP_to_Photo_Ratio": arap_ratio_log,
                "Lambda_ARAP": lambda_arap_log,
                "T_Mean": t_mean_log,
                "T_Max": t_max_log,
            })

            csv_path = os.path.join(save_path_target, "loss_log.csv")
            df.to_csv(csv_path, index=False)

            save_path_dir = os.path.join(save_path_target, f"finetuned_epoch_{epoch}")
            os.makedirs(save_path_dir, exist_ok=True)

            save_path = os.path.join(save_path_dir, f"finetuned_epoch_{epoch}.ply")

            with torch.no_grad():
                control_q_norm = torch.nn.functional.normalize(control_q, p=2, dim=-1)
                R_mat_control = quaternion_to_matrix(control_q_norm)

                current_deformed_means = deform_sam_points(
                    means,
                    control_coord,
                    R_mat_control,
                    control_T,
                    skinning_weights
                )

            save_ply_manually(
                save_path,
                current_deformed_means,
                scales,
                quats,
                opacities,
                f_dc,
                f_rest
            )

    # ----------------------------
    # Final plotting
    # ----------------------------
    final_csv = os.path.join(save_path_target, "loss_log.csv")

    if os.path.exists(final_csv):
        try:
            plot_simple_training_results(final_csv, save_path_target)
            print(f"loss log saved to: {final_csv}")
        except Exception as e:
            print(f"plot error: {e}")
            print(f"loss log saved to: {final_csv}")
    else:
        print("csv doc missing....")

    print("\nfinetune done!")


if __name__ == "__main__":
    main()