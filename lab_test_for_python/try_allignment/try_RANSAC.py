import open3d as o3d
import numpy as np
import copy
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
from plyfile import PlyData, PlyElement

# ==========================================
# 階段 1：使用 Open3D 執行 ICP 計算轉換矩陣
# ==========================================
print("\n[階段 1] 讀取點雲並執行 ICP 對齊...")
source_file = "bicycle_select_position.ply"
target_file = "bicycle_scene_selected.ply"

source = o3d.io.read_point_cloud(source_file) 
target = o3d.io.read_point_cloud(target_file) 

print(f"原始點數 -> Source: {len(source.points)}, Target: {len(target.points)}")

voxel_radius = 0.05
print(f"進行下採樣 (Voxel size: {voxel_radius})...")
source_down = source.voxel_down_sample(voxel_radius)
target_down = target.voxel_down_sample(voxel_radius)
print(f"降採樣後點數 -> Source: {len(source_down.points)}, Target: {len(target_down.points)}")
print("計算表面法向量中...")
source_down.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_radius * 2, max_nn=30))
target_down.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_radius * 2, max_nn=30))

threshold = 0.2 
trans_init = np.eye(4)

print("開始執行 Point-to-Plane ICP 對齊...")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source_down, 
    target_down, 
    threshold, 
    trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

final_transformation = reg_p2p.transformation
print("\n=== ICP 執行結果 ===")
print(f"對齊適應度 (Fitness): {reg_p2p.fitness}")
print(f"均方根誤差 (RMSE): {reg_p2p.inlier_rmse}")
print("\n最終轉換矩陣:\n", final_transformation)

# 中繼存檔 (純點雲，用於肉眼確認幾何咬合)
source_aligned = copy.deepcopy(source)
source_aligned.transform(final_transformation)
source_aligned.paint_uniform_color([1, 0, 0]) 
o3d.io.write_point_cloud("check_icp_result.ply", source_aligned)
print("✅ 中繼確認用點雲已儲存為 check_icp_result.ply")


# ==========================================
# 階段 2：直接讀取並修改 3DGS .ply 檔案參數
# ==========================================
print("\n[階段 2] 開始更新 3DGS 檔案參數 (保留所有高斯屬性)...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 使用 plyfile 讀取完整的 3DGS 資料結構
plydata = PlyData.read(source_file)
v = plydata.elements[0].data # v 是一個 Numpy Structured Array，包含所有屬性

# 2. 將 3DGS 的屬性萃取成 PyTorch Tensors 以便計算
xyz = torch.tensor(np.stack((v['x'], v['y'], v['z']), axis=-1), dtype=torch.float32, device=device)
quats = torch.tensor(np.stack((v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']), axis=-1), dtype=torch.float32, device=device)
scales = torch.tensor(np.stack((v['scale_0'], v['scale_1'], v['scale_2']), axis=-1), dtype=torch.float32, device=device)

# 3. 處理 ICP 轉換矩陣
T_matrix = torch.from_numpy(final_transformation).to(device=device, dtype=torch.float32)
linear_part = T_matrix[:3, :3]
translation = T_matrix[:3, 3]

# 提取縮放與旋轉
scale_factor = torch.norm(linear_part, dim=0).mean()
rotation_matrix = linear_part / scale_factor
R_quat = matrix_to_quaternion(rotation_matrix)

# =================更新位置 (XYZ)=================
# X' = sRX + t (這裡不需要 AABB，因為檔案裡存的已經是絕對座標)
xyz_final = (xyz @ rotation_matrix.T) * scale_factor + translation

# =================更新姿態 (Rotation)=================
R_quat_expanded = R_quat.expand(quats.shape[0], -1)
# 新旋轉 * 舊旋轉 (這裡不需要扣除 Bias，因為存檔時已經還原)
quats_final = quaternion_multiply(R_quat_expanded, quats)
# 正規化四元數以防浮點數誤差導致變形
quats_final = quats_final / quats_final.norm(dim=-1, keepdim=True)

# =================更新縮放 (Scale)=================
# 3DGS 內部存的是 log(scale)，所以用加法補償
scales_final = scales + torch.log(scale_factor)

# 4. 將算好的數值寫回 Numpy 結構陣列中
v['x'], v['y'], v['z'] = xyz_final[:,0].cpu().numpy(), xyz_final[:,1].cpu().numpy(), xyz_final[:,2].cpu().numpy()
v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3'] = quats_final[:,0].cpu().numpy(), quats_final[:,1].cpu().numpy(), quats_final[:,2].cpu().numpy(), quats_final[:,3].cpu().numpy()
v['scale_0'], v['scale_1'], v['scale_2'] = scales_final[:,0].cpu().numpy(), scales_final[:,1].cpu().numpy(), scales_final[:,2].cpu().numpy()

# 5. 使用 plyfile 寫出最終的 3DGS 檔案
PlyData([PlyElement.describe(v, 'vertex')]).write("aligned_bicycle_final_3dgs.ply")
print("🎉 恭喜！最終完美對齊且保留完整高斯屬性的 3DGS 已儲存至 aligned_bicycle_final_3dgs.ply")