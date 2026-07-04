import torch
import numpy as np
from scipy.spatial import cKDTree


@torch.no_grad()
def build_knn_graph_for_tv(
    means,
    k=8,
    use_edge_weight=True,
    eps=1e-6,
):

    device = means.device
    num_points = means.shape[0]

    if num_points <= 1:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight = None
        info = {
            "num_points": num_points,
            "actual_k": 0,
            "num_edges": 0,
        }
        return edge_index, edge_weight, info

    actual_k = min(k, num_points - 1)

    means_np = means.detach().cpu().numpy()

    tree = cKDTree(means_np)

    dists, indices = tree.query(
        means_np,
        k=actual_k + 1,
        workers=-1,
    )

    # remove self neighbor
    dists = dists[:, 1:]      # [N, k]
    indices = indices[:, 1:]  # [N, k]

    src = np.repeat(np.arange(num_points), actual_k)
    dst = indices.reshape(-1)
    edge_dists = dists.reshape(-1)

    edge_index = torch.from_numpy(
        np.stack([src, dst], axis=0)
    ).long().to(device)

    edge_dists = torch.from_numpy(edge_dists).float().to(device)

    if use_edge_weight:
        sigma = torch.median(edge_dists).clamp(min=eps)

        edge_weight = torch.exp(
            -0.5 * (edge_dists / sigma) ** 2
        ).detach()
    else:
        sigma = torch.tensor(0.0, device=device)
        edge_weight = None

    info = {
        "num_points": num_points,
        "actual_k": actual_k,
        "num_edges": edge_index.shape[1],
        "edge_dist_min": edge_dists.min().detach(),
        "edge_dist_mean": edge_dists.mean().detach(),
        "edge_dist_median": edge_dists.median().detach(),
        "edge_dist_max": edge_dists.max().detach(),
        "sigma": sigma.detach(),
    }

    return edge_index, edge_weight, info


def tv_smoothness_loss(
    membership_prob,
    edge_index,
    edge_weight=None,
    eps=1e-6,
):
    """
    Args:
        membership_prob: [N], z_i
        edge_index:      [2, E]
        edge_weight:     [E] or None

    Returns:
        loss
    """

    z = membership_prob.float()

    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=z.device)

    src = edge_index[0]
    dst = edge_index[1]

    diff = torch.abs(z[src] - z[dst])

    if edge_weight is None:
        return diff.mean()

    w = edge_weight.float().detach()

    loss = torch.sum(w * diff) / (torch.sum(w) + eps)

    return loss