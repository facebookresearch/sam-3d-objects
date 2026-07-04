import sys
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R_scipy

# 確保路徑正確
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

def get_o3d_pcd_from_gs(gs_obj):
    """將 3DGS 中心點轉為 Open3D 點雲格式"""
    xyz = gs_obj._xyz.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def execute_global_registration(source_pcd, target_pcd, voxel_size):
    """執行全域配準 (修復變數命名與參數順序)"""
    
    # 1. 下採樣與法向量估計
    print(f"-> 正在進行下採樣 (Voxel size: {voxel_size})...")
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    # 增加法向量搜尋半徑
    radius_normal = voxel_size * 5
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # 2. 提取 FPFH 特徵
    radius_feature = voxel_size * 10
    print("-> 正在提取幾何特徵 (FPFH)...")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # 3. 執行 RANSAC (嚴格遵守位置參數順序)
    distance_threshold = voxel_size * 1.5
    print("-> 正在進行全域配準 (RANSAC 碰撞中)...")
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,      # 1. Source
        target_down,      # 2. Target
        source_fpfh,      # 3. Source Feature
        target_fpfh,      # 4. Target Feature
        True,             # 5. mutual_filter
        distance_threshold, # 6. max_correspondence_distance
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), # 7. estimation_method
        4,                # 8. ransac_n
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],                # 9. checkers
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500) # 10. criteria
    )
    
    return result.transformation
def apply_transformation_to_gs(gs_obj, M):
    """將 4x4 矩陣完美套用到 3DGS (含座標、旋轉與縮放)"""
    device = gs_obj._xyz.device
    M_t = torch.tensor(M, dtype=torch.float32, device=device)
    
    # 提取旋轉矩陣 R 與平移向量 T
    R_mat = M_t[:3, :3]
    T_vec = M_t[:3, 3]
    
    # 計算平均縮放比例 (由行列式的立方根求得)
    scale_factor = torch.det(R_mat).abs().pow(1/3)
    # 正規化 R_mat 確保它是純旋轉
    R_mat_pure = R_mat / scale_factor

    with torch.no_grad():
        # 1. 更新位置: P' = R * P + T
        gs_obj._xyz.data = torch.matmul(gs_obj._xyz.data, R_mat_pure.T) * scale_factor + T_vec

        # 2. 更新旋轉 (防止刺蝟): 使用 SciPy 處理 wxyz 格式
        if hasattr(gs_obj, '_rotation'):
            rot_layout = R_scipy.from_matrix(R_mat_pure.cpu().numpy())
            quats_wxyz = gs_obj._rotation.detach().cpu().numpy()
            # 換位成 SciPy 格式 [x, y, z, w]
            quats_xyzw = np.column_stack([quats_wxyz[:, 1], quats_wxyz[:, 2], quats_wxyz[:, 3], quats_wxyz[:, 0]])
            
            new_rots = rot_layout * R_scipy.from_quat(quats_xyzw)
            new_xyzw = new_rots.as_quat()
            # 換回 3DGS 格式 [w, x, y, z]
            new_wxyz = np.column_stack([new_xyzw[:, 3], new_xyzw[:, 0], new_xyzw[:, 1], new_xyzw[:, 2]])
            gs_obj._rotation.data = torch.tensor(new_wxyz, dtype=torch.float32, device=device)

        # 3. 更新縮放
        if hasattr(gs_obj, '_scaling'):
            gs_obj._scaling.data += torch.log(scale_factor)

def run_pipeline():
    # --- Step 1: SAM3D 推理 ---
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)
    img = load_image("notebook/images/bike/original_image/image0.png")
    msk = load_single_mask("notebook/images/bike/samhq_image", index=0)
    outputs = inference(img, msk, seed=42)
    gs_obj = outputs['gs']

    # --- Step 2: 自動對齊 ---
    colmap_pcd_path = "/work/goet1019/dataset/3DGS_output/bicycle/point_cloud/iteration_30000/bicycle_scene_selected.ply" # <-- 請填入你的真實場景點雲路徑
    source_pcd = get_o3d_pcd_from_gs(gs_obj)
    target_pcd = o3d.io.read_point_cloud(colmap_pcd_path)
    
    # 執行配準拿到 M
    M = execute_global_registration(source_pcd, target_pcd, voxel_size=0.05)

    # --- Step 3: 套用並儲存 ---
    apply_transformation_to_gs(gs_obj, M)
    gs_obj.save_ply("aligned_bike_in_scene.ply")
    print("\n=== 任務完成！請在 SuperSplat 同時打開場景與 aligned_bike_in_scene.ply ===")

if __name__ == "__main__":
    run_pipeline()