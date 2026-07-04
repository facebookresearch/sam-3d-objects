import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


@torch.no_grad()
def compute_nearest_distance_to_sam3d_pointcloud(
    means,
    sam3d_points,
):
    """
    Args:
        means:        [N, 3], cropped Gaussian centers
        sam3d_points: [M, 3], aligned SAM3D point cloud

    Returns:
        sam3d_dist: [N], nearest distance to SAM3D point cloud
    """

    device = means.device

    means_np = means.detach().cpu().numpy()
    sam3d_np = sam3d_points.detach().cpu().numpy()

    tree = cKDTree(sam3d_np)

    dists, _ = tree.query(
        means_np,
        k=1,
        workers=-1,
    )

    sam3d_dist = torch.from_numpy(dists).float().to(device)

    return sam3d_dist


@torch.no_grad()
def build_sam3d_soft_target_from_pointcloud(
    means,
    sam3d_points,
    margin_ratio=0.01,
    sigma_ratio=0.03,
    eps=1e-6,
):
    """
    Build SAM3D soft target q_i for each cropped Gaussian.

    First version:
        Use SAM3D point cloud as a soft volume prior.

    Args:
        means:        [N, 3], cropped Gaussian centers
        sam3d_points: [M, 3], aligned SAM3D point cloud

    Returns:
        sam3d_target: [N], q_i in [0, 1]
        sam3d_dist:   [N], nearest distance to SAM3D point cloud
        info: dict
    """

    sam3d_dist = compute_nearest_distance_to_sam3d_pointcloud(
        means=means,
        sam3d_points=sam3d_points,
    )

    bbox_min = sam3d_points.min(dim=0).values
    bbox_max = sam3d_points.max(dim=0).values
    bbox_diag = torch.norm(bbox_max - bbox_min).clamp(min=eps)

    margin = margin_ratio * bbox_diag
    sigma = sigma_ratio * bbox_diag
    sigma = sigma.clamp(min=eps)

    # d_out = max(d - margin, 0)
    # within margin: q_i = 1
    outside_dist = torch.relu(sam3d_dist - margin)

    sam3d_target = torch.exp(
        -0.5 * (outside_dist / sigma) ** 2
    )

    sam3d_target = sam3d_target.clamp(0.0, 1.0)

    info = {
        "bbox_diag": bbox_diag.detach(),
        "margin": margin.detach(),
        "sigma": sigma.detach(),
        "dist_min": sam3d_dist.min().detach(),
        "dist_mean": sam3d_dist.mean().detach(),
        "dist_median": sam3d_dist.median().detach(),
        "dist_max": sam3d_dist.max().detach(),
        "q_mean": sam3d_target.mean().detach(),
        "q_gt_05_ratio": (sam3d_target > 0.5).float().mean().detach(),
        "q_gt_01_ratio": (sam3d_target > 0.1).float().mean().detach(),
    }

    return sam3d_target.detach(), sam3d_dist.detach(), info

def sam3d_prior_loss(
    membership_prob,
    sam3d_target,
    neg_weight=0.25,
    eps=1e-6,
):
    """
    Args:
        membership_prob: [N], z_i = sigmoid(c_i)
        sam3d_target:    [N], q_i from SAM3D soft prior
        neg_weight:      negative penalty weight

    Returns:
        loss
    """

    z = membership_prob.float().clamp(eps, 1.0 - eps)
    q = sam3d_target.float().detach().clamp(0.0, 1.0)

    pos_loss = q * (-torch.log(z))
    neg_loss = (1.0 - q) * (-torch.log(1.0 - z))

    loss = pos_loss + neg_weight * neg_loss

    normalizer = q.sum() + neg_weight * (1.0 - q).sum() + eps

    return loss.sum() / normalizer