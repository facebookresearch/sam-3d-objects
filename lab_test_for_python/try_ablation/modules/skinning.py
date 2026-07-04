import torch
import torch.nn.functional as F

def compute_knn_rbf_weights(gauss_means, control_p, control_o, k=5):
    """
    計算每個高斯點受 KNN 控制點影響的 RBF 權重。
    參數:
        gauss_means: [M, 3] SAM3D 點雲的座標
        control_p:   [N, 3] 控制點的座標
        control_o:   [N, 1] 控制點的原始 RBF 半徑
        k:           整數，每個點最多受幾個控制點影響 (推薦 3~5)
        
    回傳:
        dense_weights: [M, N] 正規化後的權重矩陣，非 KNN 範圍內的權重為 0
    """
    M = gauss_means.shape[0]
    N = control_p.shape[0]
    dist_sq = torch.cdist(gauss_means, control_p) ** 2
    knn_dist_sq, knn_indices = torch.topk(dist_sq, k, dim=1, largest=False)
    safe_o = F.softplus(control_o) 
    knn_o = safe_o.view(-1)[knn_indices] 
    rbf_values = torch.exp(-knn_dist_sq / (2 * knn_o ** 2 + 1e-8)) 
    weights_sum = rbf_values.sum(dim=1, keepdim=True) + 1e-8
    normalized_weights = rbf_values / weights_sum 
    dense_weights = torch.zeros((M, N), device=gauss_means.device)
    dense_weights.scatter_(1, knn_indices, normalized_weights)

    return dense_weights


def deform_sam_points(sam_points, p, R_mat, T, weights):
    """
    sam_points: [M, 3]
    p: [N, 3] 
    R_mat: [N, 3, 3]
    T: [N, 3]
    weights: [M, N]
    """

    M = sam_points.shape[0]
    N = p.shape[0]
    deformed_points = torch.zeros_like(sam_points)
    
    for i in range(N):
        w_i = weights[:, i].unsqueeze(1) # [M, 1]
        centered_pts = sam_points - p[i]
        rotated_pts = torch.matmul(centered_pts, R_mat[i].T)
        transformed_pts = rotated_pts + p[i] + T[i]
        deformed_points += w_i * transformed_pts
        
    return deformed_points