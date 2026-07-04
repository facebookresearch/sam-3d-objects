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

from data_preprocessing import BicycleFinetuneDataset
from util import (
    render_standalone, 
    load_ply_manually, 
    save_ply_manually, 
    get_ssim_loss,
    get_l1_loss,
    plot_training_results,
    initialize_control_points
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
    
    scales = nn.Parameter(s.requires_grad_(True))
    quats = nn.Parameter(q.requires_grad_(True))
    opacities = nn.Parameter(o.requires_grad_(True))
    f_dc = nn.Parameter(f_dc_init.requires_grad_(False))
    

    f_rest = f_rest_init.requires_grad_(False)
    with torch.no_grad():
        initial_physical_scales = torch.exp(s)
        base_scale = torch.quantile(initial_physical_scales.view(-1), 0.95).item()
        
        max_raw_scale = math.log(base_scale * 5.0)
        
        print(f"the maximum scale in training is set to {max_raw_scale:.4f} (base scale: {base_scale:.4f})")

    optimizer = torch.optim.Adam([
        {'params': [means], 'lr': args.position_lr, "name": "xyz"},
        #{'params': [f_dc], 'lr': args.feature_lr, "name": "f_dc"},
        {'params': [opacities], 'lr': args.opacity_lr, "name": "opacity"},
        {'params': [scales], 'lr': args.scaling_lr, "name": "scaling"},
        {'params': [quats], 'lr': args.rotation_lr, "name": "rotation"}
    ], lr=0.0, eps=1e-15)
    
    total_epochs = args.iterations // len(dataset)     
    print(f"total {total_epochs} epochs...")
    progress_bar = tqdm(range(1,total_epochs+1), desc="Epoch Progress")
    save_path_target = os.path.join(args.output_dir, f"L1_plus_SSIM_loss_no_sh_scale_restrict_with_remove_bad_photo_opcity_restriction_0.2")
    os.makedirs(save_path_target, exist_ok=True)
    for epoch in progress_bar:
        epoch_ssim_loss_sum = 0.0
        epocH_l1_loss_sum=0.0
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
            
            actual_scales = torch.exp(scales)
            actual_opacities = torch.sigmoid(opacities)
            norm_quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
            
            colors = 0.5 + 0.28209 * f_dc


            render_out = render_standalone(means, actual_scales, norm_quats, actual_opacities, colors, camera_dict)
            
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

            #restrict the scale to prevent training collapse
            with torch.no_grad():
                scales.clamp_(max=max_raw_scale)

            epoch_ssim_loss_sum += loss_ssim.item()
            epocH_l1_loss_sum+=loss_L1.item()

        epoch_avg_ssim_loss = epoch_ssim_loss_sum / len(dataset)
        epoch_avg_l1_loss = epocH_l1_loss_sum / len(dataset)
        epoch_avg_total_loss = epoch_total_loss_sum / len(dataset)

        ssim_loss_log.append(epoch_avg_ssim_loss)
        l1_loss_log.append(epoch_avg_l1_loss)
        total_loss_log.append(epoch_avg_total_loss)
        progress_bar.set_postfix({"Total_Loss": f"{epoch_avg_total_loss:.4f}"})
           
        if epoch % 10 == 0 or epoch == total_epochs:
            with torch.no_grad():
                actual_opacities_prune = torch.sigmoid(opacities).squeeze()
                
                PRUNE_THRESHOLD = 0.2 
                keep_mask = actual_opacities_prune > PRUNE_THRESHOLD
                
                num_before = len(actual_opacities_prune)
                num_after = keep_mask.sum().item()
                num_deleted = num_before - num_after
                
                if num_deleted > 0:
                    print(f"[Epoch {epoch}] pruned:delete {num_deleted} gaussians (remaining {num_after} )。")
                    
                    non_optim_params = {"f_dc": f_dc, "f_rest": f_rest}
                    
                    updated_params = prune_gaussians(keep_mask, optimizer, non_optim_params)
                    
                    means = updated_params["xyz"]
                    opacities = updated_params["opacity"]
                    scales = updated_params["scaling"]
                    quats = updated_params["rotation"]
                    f_dc = updated_params["f_dc"]
                    f_rest = updated_params["f_rest"]
                    norm_quats = torch.nn.functional.normalize(quats, p=2, dim=-1)


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
            save_ply_manually(save_path, means, scales, norm_quats, opacities, f_dc, f_rest)

            if epoch >= 20 and epoch != total_epochs:
                print(f"\n[Epoch {epoch}] evaluating dataset for extreme outliers...")
                
                eval_losses = {}
                
                with torch.no_grad():
                    for i in range(len(dataset)):
                        camera_dict, gt_image, gt_mask = dataset[i]
                        gt_image = gt_image.squeeze(0).to(device)  # [3, H, W]
                        gt_mask = gt_mask.squeeze(0).to(device)    # [1, H, W]
                        photo_name = camera_dict["image_name"]
                        masked_gt = gt_image * gt_mask
                        
                        actual_scales = torch.exp(scales)
                        actual_opacities = torch.sigmoid(opacities)
                        norm_quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
                        
                        colors = 0.5 + 0.28209 * f_dc


                        render_out = render_standalone(means, actual_scales, norm_quats, actual_opacities, colors, camera_dict)
                        
                        render_out = render_out.permute(2, 0, 1)
                        render_out = render_out.clamp(0.0, 1.0)

                        masked_rendered = render_out * gt_mask

                        loss_L1=get_l1_loss(masked_rendered, masked_gt)
                        loss_ssim = get_ssim_loss(masked_rendered, masked_gt)


                        total_loss = (1 - args.lambda_dssim) * loss_L1 + args.lambda_dssim * loss_ssim
                        
                        eval_losses[photo_name] = loss_L1.item()
                
                all_loss_vals = np.array(list(eval_losses.values()))
                mean_loss = all_loss_vals.mean()
                std_loss = all_loss_vals.std()
                dynamic_threshold = mean_loss + 2.0 * std_loss
                min_loss_floor = 0.020
                threshold = max(dynamic_threshold, min_loss_floor)
                bad_photos = [name for name, loss in eval_losses.items() if loss > threshold]
                
                if len(bad_photos) > 0:
                    print(f"find {len(bad_photos)} bad photos (Threshold: {threshold:.4f})")
                    deleted_info=[]
                    for bad_p in bad_photos:
                        print(f"    delete: {bad_p} (Loss: {eval_losses[bad_p]:.4f})")
                        deleted_info.append({
                            "Photo_Name": bad_p,
                            "Loss": eval_losses[bad_p],
                            "Threshold": threshold
                        })

                    log_csv_path = os.path.join(save_path_dir, "deleted_photos_history.csv")
                    df_deleted = pd.DataFrame(deleted_info)
                    df_deleted.to_csv(log_csv_path, index=False)

                    dataset.remove_bad_data(bad_photos)
                    
                    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                    dataloader_iterator = iter(dataloader)
                    print(f"DataLoader reset, remaining photos: {len(dataset)}")

    last_csv = os.path.join(save_path_target, f"finetuned_epoch_{total_epochs}/loss_log_iter_{total_epochs}.csv")
    if  os.path.exists(last_csv):
        try:
            plot_training_results(last_csv, save_path_target)
        except Exception as e:
            print(f"plot error{e}") 
    else:
        print("csv doc missing....")
    # try:
    #     last_csv = os.path.join(save_path_target, f"finetuned_epoch_{total_epochs}/loss_log_iter_{total_epochs}.csv")
    #     print(last_csv)
    #     plot_training_results(last_csv, save_path_target)
    # except Exception as e:
    #     print(f"cant plot the diagram...,maybe u dont train entire round")
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
    
    print(f"📊 原始點雲中心: {means.mean(0).tolist()}")

    print("📸 執行原生 gsplat 渲染...")
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
        
        print("✅ 實驗完成！已輸出 STANDALONE_TEST.png")
        print(f"🖼️ 圖片尺寸: {render_out.shape[1]}x{render_out.shape[2]}")
        
    except Exception as e:
        import traceback
        traceback.print_exc() 
        print(f"❌ 渲染失敗: {e}")

if __name__ == "__main__":
    run_standalone_experiment()
'''

'''
def test_ply_io():
    input_path = "../some_basic_pointcloud/mip_nerf_reset_selected_scene.ply" 
    output_path = "./DEBUG_TEST_SAVE_FULL.ply"

    print("========================================")
    print("🚀 開始測試「完整 SH」讀寫管線")
    print("========================================")

    try:
        # --- 步驟 A: 測試讀取 (解構 6 個變數) ---
        print(f"\n[A] 正在讀取: {input_path}")
        means, scales, quats, opacities, f_dc, f_rest = load_ply_manually(input_path)
        print(f"✅ 讀取成功！點雲數量: {means.shape[0]}")
        print(f"📊 SH 結構檢查: DC={f_dc.shape}, Rest={f_rest.shape}")

        # --- 步驟 B: 測試儲存 (傳入 6 個變數) ---
        print(f"\n[B] 正在儲存至: {output_path}")
        save_ply_manually(output_path, means, scales, quats, opacities, f_dc, f_rest)
        print(f"✅ 儲存成功！")

        # --- 步驟 C: 檔案檢查 ---
        if os.path.exists(output_path):
            original_size = os.path.getsize(input_path) / (1024 * 1024)
            new_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n[C] 檔案大小比對:")
            print(f"📦 原始檔案: {original_size:.2f} MB")
            print(f"📦 新存檔案: {new_size:.2f} MB")
            
            # 這次誤差應該要縮小到 1MB 以內，因為欄位全對齊了
            if abs(original_size - new_size) < 1.0:
                print("🎉 完美！檔案大小幾乎完全一致，SH 數據無損！")
            else:
                print("⚠️ 警告：檔案大小仍有顯著差異，請確認欄位順序。")
        else:
            print("❌ 錯誤：找不到儲存的檔案！")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 測試過程發生錯誤: {e}")
        '''