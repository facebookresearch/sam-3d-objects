import torch
from sklearn.neighbors import NearestNeighbors

def fps(points, points_num):

    device = points.device
    N, D = points.shape
    xyz = points[:, :3]
    
    centroids = torch.zeros(points_num, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)
    
    for i in range(points_num):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return centroids 

def get_init_o_by_knn(points, k=3):
    points_np = points.detach().cpu().numpy()

    nn_model = NearestNeighbors(n_neighbors=k + 1).fit(points_np)
    distances, _ = nn_model.kneighbors(points_np)

    avg_dist = distances[:, 1:].mean(axis=1)

    return torch.from_numpy(avg_dist).float().to(points.device).unsqueeze(-1)


def initialize_control_points(means, num_points, k_neighbors=3):
    if means.shape[0] <= num_points:
        print(f"Number of points ({means.shape[0]}) <= desired {num_points}, using all points.")
        selected_p = means.clone()
    else:
        selected_indices = fps(means, num_points)
        selected_p = means[selected_indices]
    
    selected_o = get_init_o_by_knn(selected_p, k=k_neighbors)
    
    return selected_p, selected_o