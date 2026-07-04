import torch

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

def compute_knn_edge_weights(edge_index, edge_dist, num_points, sigma=None, eps=1e-8):
    """
    Compute normalized RBF weights for directed control-control edges.

    For each source point i:
        sum_k w_ik ? 1

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
