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
import torchvision.transforms.functional as TF
from PIL import Image
from plyfile import PlyData
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
    plot_training_results,
    initialize_control_points,
    compute_knn_rbf_weights,
    deform_sam_points
)

def prune_gaussians(keep_mask, optimizer, non_optim_params):
    new_params = {}
    for group in optimizer.param_groups:
        p = group["params"][0]
        name = group["name"]
        stored_state = optimizer.state.get(p, None)
        
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][keep_mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][keep_mask]
            del optimizer.state[p] 
        
        new_p = nn.Parameter(p[keep_mask].detach().requires_grad_(True))
        group["params"][0] = new_p
        
        if stored_state is not None:
            optimizer.state[new_p] = stored_state
            
        new_params[name] = new_p
        
    if "f_dc" in non_optim_params:
        new_params["f_dc"] = nn.Parameter(non_optim_params["f_dc"][keep_mask].detach(), requires_grad=False)
    if "f_rest" in non_optim_params:
        new_params["f_rest"] = nn.Parameter(non_optim_params["f_rest"][keep_mask].detach(), requires_grad=False)
        
    return new_params
def parse_args():
    parser = argparse.ArgumentParser(description="3DGS Bicycle Fine-tuning Transparently")
    parser.add_argument("--data_dir", type=str, default="/work/goet1019/dataset/mip_nerf360/3dgs_camera_pose/bicycle")
    parser.add_argument("--initial_sam3d_ply_path", default="/work/goet1019/sam-3d-objects/lab_test_for_python/some_basic_pointcloud/bicycle_select_position.ply",type=str, help="Path to SAM3D PLY")
    parser.add_argument("--output_dir", type=str, default="./output/bicycle_finetuned")
    parser.add_argument("--iterations", type=int, default=19400)
    parser.add_argument("--position_lr", type=float, default=0.0016)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 3000])
    parser.add_argument("--control_point_r_lr",type=float, default=0.0005)
    parser.add_argument("--control_point_coord_lr",type=float, default=0.0001)
    parser.add_argument("--control_point_rbf_lr",type=float, default=0.0001)
    parser.add_argument("--control_point_t_lr", type=float, default=0.001)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")
    
    dataset = BicycleFinetuneDataset(data_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader_iterator = iter(dataloader)
    ssim_loss_log=[]
    l1_loss_log=[]
    total_loss_log=[]
    print("load ply data...")

    m, s, q, o, f_dc_init, f_rest_init = load_ply_manually(args.initial_sam3d_ply_path)
    c_coord,c_rbf=initialize_control_points(m,num_points=512)


    #stage 1:train translation and rotation 
    control_T = nn.Parameter(torch.zeros_like(c_coord), requires_grad=True)
    init_q = torch.zeros(c_coord.shape[0], 4, device=c_coord.device)
    init_q[:, 0] = 1.0
    control_q = nn.Parameter(init_q, requires_grad=False)

    #stage 2:control point coord and radii 
    control_coord = nn.Parameter(c_coord.detach(), requires_grad=False)
    control_rbf = nn.Parameter(c_rbf.detach(), requires_grad=False)

    #stage 3:3DGS attribute
    means = nn.Parameter(m.detach(), requires_grad=False)
    scales = nn.Parameter(s.detach(), requires_grad=False)
    quats = nn.Parameter(q.detach(), requires_grad=False)
    opacities = nn.Parameter(o.detach(), requires_grad=False)
    f_dc = nn.Parameter(f_dc_init.detach(), requires_grad=False)
    
    f_rest = nn.Parameter(f_rest_init.detach(), requires_grad=False)
    

    with torch.no_grad():
        initial_physical_scales = torch.exp(s)
        base_scale = torch.quantile(initial_physical_scales.view(-1), 0.95).item()
        
        max_raw_scale = math.log(base_scale * 5.0)
        
        print(f"the maximum scale in training is set to {max_raw_scale:.4f} (base scale: {base_scale:.4f})")

    optimizer = torch.optim.Adam([
        {'params':[control_T], 'lr': args.control_point_t_lr, "name": "control_T"}
        #{'params':[control_q], 'lr': args.control_point_r_lr, "name": "control_q"}
        
        #{'params':[control_coord], 'lr': args.control_point_coord_lr, "name": "control_coord"},
        #{'params':[control_rbf], 'lr': args.control_point_rbf_lr, "name": "control_rbf"},
        #{'params': [means], 'lr': args.position_lr, "name": "xyz"},
        #{'params': [f_dc], 'lr': args.feature_lr, "name": "f_dc"},
        #{'params': [opacities], 'lr': args.opacity_lr, "name": "opacity"},
        #{'params': [scales], 'lr': args.scaling_lr, "name": "scaling"},
        #{'params': [quats], 'lr': args.rotation_lr, "name": "rotation"}
        
    ], lr=0.0, eps=1e-15)
    
    total_epochs = args.iterations // len(dataset)     
    print(f"total {total_epochs} epochs...")
    print(f"stage 1 epch: {total_epochs//2}, stage 2 epoch: {total_epochs//3}, stage 3 epoch: {total_epochs - total_epochs//2 - total_epochs//3}")
    print(f"stage 1: training control point translation and rotation...")
    stage_1_epochs = total_epochs // 2
    stage_2_epochs = total_epochs // 3
    stage_3_epochs = total_epochs - stage_1_epochs - stage_2_epochs

    progress_bar = tqdm(range(1,stage_1_epochs+1), desc="Epoch Progress")
    save_path_target = os.path.join(args.output_dir, f"L1_plus_SSIM_loss_Stage1_no_rotation")
    os.makedirs(save_path_target, exist_ok=True)

    print("計算初始 RBF 權重...")
    skinning_weights = compute_knn_rbf_weights(means, control_coord, control_rbf).detach()

    with torch.no_grad():
        static_actual_scales = torch.exp(scales)
        static_norm_quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
        static_actual_opacities = torch.sigmoid(opacities)
        static_colors = 0.5 + 0.28209 * f_dc  
        control_q_norm = torch.nn.functional.normalize(control_q, dim=-1)
        static_R_mat_control = quaternion_to_matrix(control_q_norm)

    for epoch in progress_bar:
        epoch_ssim_loss_sum = 0.0
        epoch_l1_loss_sum=0.0
        epoch_total_loss_sum=0.0
        
        for each_step in range(len(dataset)):
            try:
                camera_dict, gt_image, gt_mask= next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                camera_dict, gt_image, gt_mask = next(dataloader_iterator)
            gt_image = gt_image.squeeze(0).to(device)  # [3, H, W]
            gt_mask = gt_mask.squeeze(0).to(device)    # [1, H, W]
            
            masked_gt = gt_image * gt_mask



            deformed_means = deform_sam_points(means, control_coord, static_R_mat_control, control_T, skinning_weights)
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

            loss_L1=get_l1_loss(masked_rendered, masked_gt)
            loss_ssim = get_ssim_loss(masked_rendered, masked_gt)

            total_loss = (1 - args.lambda_dssim) * loss_L1 + args.lambda_dssim * loss_ssim
            epoch_total_loss_sum += total_loss.item()
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_ssim_loss_sum += loss_ssim.item()
            epoch_l1_loss_sum+=loss_L1.item()

        epoch_avg_ssim_loss = epoch_ssim_loss_sum / len(dataset)
        epoch_avg_l1_loss = epoch_l1_loss_sum / len(dataset)
        epoch_avg_total_loss = epoch_total_loss_sum / len(dataset)

        ssim_loss_log.append(epoch_avg_ssim_loss)
        l1_loss_log.append(epoch_avg_l1_loss)
        total_loss_log.append(epoch_avg_total_loss)
        progress_bar.set_postfix({"Total_Loss": f"{epoch_avg_total_loss:.4f}"})
           
        if epoch % 10 == 0 or epoch == stage_1_epochs:
            save_path_dir = os.path.join(save_path_target, f"finetuned_epoch_{epoch}")
            os.makedirs(save_path_dir, exist_ok=True)
            save_path = os.path.join(save_path_dir, f"finetuned_epoch_{epoch}.ply")

            df = pd.DataFrame({
                "Iteration": range(1, len(total_loss_log) + 1),
                "SSIM_Loss": ssim_loss_log,
                "L1_Loss": l1_loss_log,
                "Total_Loss": total_loss_log
            })
            csv_path = os.path.join(save_path_dir, f"loss_log_iter_{epoch}.csv")
            df.to_csv(csv_path, index=False)
            with torch.no_grad():
                current_deformed_means = deform_sam_points(means, control_coord, static_R_mat_control, control_T, skinning_weights)
            save_ply_manually(save_path, current_deformed_means, scales, quats, opacities, f_dc, f_rest)

    last_csv = os.path.join(save_path_target, f"finetuned_epoch_{stage_1_epochs}/loss_log_iter_{stage_1_epochs}.csv")
    if  os.path.exists(last_csv):
        try:
            plot_training_results(last_csv, save_path_target)
        except Exception as e:
            print(f"plot error{e}") 
    else:
        print("csv doc missing....")
    print("\nfinetune done!")

if __name__ == "__main__":
    main()

'''
def run_standalone_experiment():
    device = torch.device("cuda")
    
    # 1. 載入 Dataset
    dataset = BicycleFinetuneDataset(data_dir="/work/goet1019/dataset/mip_nerf360/3dgs_camera_pose/bicycle")
    camera_dict, gt_image, gt_mask,masked_img_tensor = dataset[1] 
    
    # 2. 手動載入 PLY
    model_path = "../some_basic_pointcloud/mip_nerf_reset_selected_scene.ply"
    means, scales, quats, opacities, colors = load_ply_manually(model_path)
    
    print(f"? 原始點雲中心: {means.mean(0).tolist()}")

    print("? 執行原生 gsplat 渲染...")
    try:

        render_out = render_standalone(means, scales, quats, opacities, colors, camera_dict)
        
        # 4. 處理輸出 [H, W, 3] -> [3, H, W]
        # 安全檢查：確保 render_out 在 GPU 上，並做數值截斷 (Clamp)
        render_out = render_out.detach().clamp(0.0, 1.0) 

        if render_out.shape[-1] == 3:
            render_out = render_out.permute(2, 0, 1)
            
        # 5. 存檔
        # 使用 .cpu() 確保資料回到記憶體再轉 PIL
        out_img = TF.to_pil_image(render_out.cpu())
        out_img.save("STANDALONE_TEST.png")
        
        save_mask_img = TF.to_pil_image(masked_img_tensor)
        save_mask_img.save(f"test_output_mask.png")

        # 存對照組時也做一下 clamp，防止 GT 數值異常
        gt_img_pil = TF.to_pil_image(gt_image.clamp(0.0, 1.0).cpu())
        gt_img_pil.save("STANDALONE_GT.png")
        
        print("? 實驗完成！已輸出 STANDALONE_TEST.png")
        print(f"?? 圖片尺寸: {render_out.shape[1]}x{render_out.shape[2]}")
        
    except Exception as e:
        import traceback
        traceback.print_exc() 
        print(f"? 渲染失敗: {e}")

if __name__ == "__main__":
    run_standalone_experiment()
'''

'''
def test_ply_io():
    input_path = "../some_basic_pointcloud/mip_nerf_reset_selected_scene.ply" 
    output_path = "./DEBUG_TEST_SAVE_FULL.ply"

    print("========================================")
    print("? 開始測試「完整 SH」讀寫管線")
    print("========================================")

    try:
        # --- 步驟 A: 測試讀取 (解構 6 個變數) ---
        print(f"\n[A] 正在讀取: {input_path}")
        means, scales, quats, opacities, f_dc, f_rest = load_ply_manually(input_path)
        print(f"? 讀取成功！點雲數量: {means.shape[0]}")
        print(f"? SH 結構檢查: DC={f_dc.shape}, Rest={f_rest.shape}")

        # --- 步驟 B: 測試儲存 (傳入 6 個變數) ---
        print(f"\n[B] 正在儲存至: {output_path}")
        save_ply_manually(output_path, means, scales, quats, opacities, f_dc, f_rest)
        print(f"? 儲存成功！")

        # --- 步驟 C: 檔案檢查 ---
        if os.path.exists(output_path):
            original_size = os.path.getsize(input_path) / (1024 * 1024)
            new_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n[C] 檔案大小比對:")
            print(f"? 原始檔案: {original_size:.2f} MB")
            print(f"? 新存檔案: {new_size:.2f} MB")
            
            # 這次誤差應該要縮小到 1MB 以內，因為欄位全對齊了
            if abs(original_size - new_size) < 1.0:
                print("? 完美！檔案大小幾乎完全一致，SH 數據無損！")
            else:
                print("?? 警告：檔案大小仍有顯著差異，請確認欄位順序。")
        else:
            print("? 錯誤：找不到儲存的檔案！")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n? 測試過程發生錯誤: {e}")
        '''