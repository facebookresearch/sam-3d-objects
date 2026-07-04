import struct
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import numpy as np
from plyfile import PlyData,PlyElement
from PIL import Image
import math
import gsplat
from gsplat import rasterization
from gsplat import proj
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
# 這裡是解析 COLMAP 的 binary 格式，轉換成我們需要的相機參數格式
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        #print(f"Found {num_cameras} cameras in {path_to_model_file}")
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1] 
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = 4 if model_id == 1 else 3 # SIMPLE_PINHOLE 有 3 個

            params = read_next_bytes(fid, num_params * 8, "d" * num_params)
            cameras[camera_id] = {
                "model": model_id, "width": width, "height": height, "params": np.array(params)
            }
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        #print(f"Found {num_images} images in {path_to_model_file}")
        for _ in range(num_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5]) # 四元數 [w, x, y, z]
            tvec = np.array(binary_image_properties[5:8]) # 平移向量 [tx, ty, tz]
            camera_id = binary_image_properties[8]

            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            read_next_bytes(fid, num_points2D * 24, "ddq" * num_points2D)
            
            images[image_id] = {
                "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": image_name
            }
    return images
#結束


#render
def load_ply_manually(path):
    print(f"[PLY] loading ply....: {path}")
    plydata = PlyData.read(path)
    v = plydata['vertex']
    
    means = torch.from_numpy(np.stack([v['x'], v['y'], v['z']], axis=-1)).float().cuda()
    scales = torch.from_numpy(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1)).float().cuda()
    quats = torch.from_numpy(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1)).float().cuda()
    opacities = torch.from_numpy(v['opacity']).float().cuda()
    
    f_dc = torch.from_numpy(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1)).float().cuda()
    existing_names = [p.name for p in plydata.elements[0].properties]
    if "f_rest_0" in existing_names:
        f_rest_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        f_rest_names = sorted(f_rest_names, key=lambda x: int(x.split('_')[-1]))
        f_rest = torch.from_numpy(np.stack([v[name] for name in f_rest_names], axis=-1)).float().cuda()
    else:
        print("f_rest attribute not found, initializing to zeros....")
        num_pts = means.shape[0]
        f_rest = torch.zeros((num_pts, 45), device="cuda").float()

    print(f"loaded PLY successfully!")
    
    return means, scales, quats, opacities, f_dc, f_rest

def render_standalone(means, scales, quats, opacities, colors, camera_dict):
    device = means.device
    W, H = int(camera_dict["width"]), int(camera_dict["height"])
    
    fov_x = camera_dict["FovX"]
    fov_y = camera_dict["FovY"]

    if fov_x > 10: 
        fov_x = math.radians(fov_x)
        fov_y = math.radians(fov_y)

    fx = W / (2 * math.tan(fov_x / 2))
    fy = H / (2 * math.tan(fov_y / 2))
    cx, cy = W / 2.0, H / 2.0

    Ks = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], device=device).float()

    R = camera_dict["R"].float().to(device) 
    T = camera_dict["T"].float().to(device) 
    
    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = T
    viewmats = viewmat.unsqueeze(0)
    render_colors, _, meta = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities.squeeze(-1) if opacities.dim() > 1 else opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        near_plane=0.01,
        far_plane=1000.0,
        render_mode="RGB"
    )



    return render_colors.squeeze(0)


def save_ply_manually(path, means, scales, quats, opacities, f_dc, f_rest=None):

    print(f"saving PLY to {path}...")
    
    xyz = means.detach().cpu().numpy()
    normals = np.zeros_like(xyz)      #目前沒有用到法向量，填0
    f_dc_np = f_dc.detach().cpu().numpy()
    

    opacities_np = opacities.detach().cpu().numpy()
    if opacities_np.ndim == 1: 
        opacities_np = opacities_np[:, None]
        
    scale_np = scales.detach().cpu().numpy()
    
    rotation_np = quats.detach().cpu().numpy()

    attributes_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    
    for i in range(3):
        attributes_names.append(f'f_dc_{i}')
        

    if f_rest is not None:
        f_rest_np = f_rest.detach().cpu().numpy()
        for i in range(f_rest_np.shape[1]):
            attributes_names.append(f'f_rest_{i}')
            
    attributes_names.extend(['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])


    dtype_full = [(attr, 'f4') for attr in attributes_names]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    
   
    if f_rest is not None:
        attributes = np.concatenate((xyz, normals, f_dc_np, f_rest_np, opacities_np, scale_np, rotation_np), axis=1)
    else:
        attributes = np.concatenate((xyz, normals, f_dc_np, opacities_np, scale_np, rotation_np), axis=1)
        
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    
    # 4. 強制寫入 Binary Little Endian
    with open(path, 'wb') as f:
        PlyData([el], text=False, byte_order='<').write(f)
        
    print(f"saved PLY to {path} successfully!")
#render結束

#loss function
def get_ssim_loss(rendered_image, gt_image):

    img1 = rendered_image.unsqueeze(0)
    img2 = gt_image.unsqueeze(0)

    ssim_value = ssim(img1, img2, data_range=1.0, size_average=True)

    return 1.0 - ssim_value

def get_l1_loss(rendered_image, gt_image):
    return torch.nn.functional.l1_loss(rendered_image, gt_image)

#loss function end

#plot_diagram
def plot_training_results(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    x_axis = "Iteration" if "Iteration" in df.columns else "Epoch"
    if x_axis not in df.columns:
        df = df.reset_index().rename(columns={'index': 'Step'})
        x_axis = 'Step'
    metrics = [col for col in df.columns if col != x_axis]

    plt.figure(figsize=(10, 6), dpi=150)
    
    for metric in metrics:
        plt.plot(df[x_axis], df[metric], label=metric, linewidth=1.5)

    plt.title("Training Loss Curve", fontsize=14)
    plt.xlabel(x_axis, fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = os.path.join(output_dir, "training_plot.png")
    plt.savefig(save_path)
    plt.close()


# control point FPS
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
    nn_model = NearestNeighbors(n_neighbors=k+1).fit(points_np)
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
'''
if __name__ == "__main__":
    m, s, q, o, f_dc_init, f_rest_init = load_ply_manually("/work/goet1019/sam-3d-objects/lab_test_for_python/some_basic_pointcloud/bicycle_select_position.ply")
    control_coords,control_RBF=initialize_control_points(m, num_points=1000, k_neighbors=3)
    control_point=torch.cat([control_coords, control_RBF], dim=-1)
    print(control_point[:5,:])
'''