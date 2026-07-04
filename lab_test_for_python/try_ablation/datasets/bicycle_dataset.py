import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import math
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from pytorch3d.transforms import quaternion_to_matrix

from datasets.colmap_loader import read_cameras_binary, read_images_binary

class BicycleFinetuneDataset(Dataset):
    def __init__( self,data_dir,image_folder="images_4",mask_folder="masks",sparse_folder="sparse/0",split="train",):
        super().__init__()
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_folder)
        self.mask_dir = os.path.join(data_dir, mask_folder)
        self.sparse_dir = os.path.join(data_dir, sparse_folder)
        self.image_filenames = sorted(os.listdir(self.image_dir))

        self.read_colmap_data(self.sparse_dir)
        print(f"成功載入 {len(self.image_filenames)} 筆 {split} 資料！")
        
    def read_colmap_data(self, sparse_path):
        images_data = read_images_binary(os.path.join(sparse_path, "images.bin"))
        cameras_data = read_cameras_binary(os.path.join(sparse_path, "cameras.bin"))

        self.poses_dict = {}
        for img_id, img in images_data.items():
            cam = cameras_data[img["camera_id"]]
            width = 1237
            height = 822

            scale = width / cam["width"]
        
            
            model_id = cam["model"]

            if model_id == 0: 
                focal_x = cam["params"][0] * scale
                focal_y = focal_x
                c_x = cam["params"][1] * scale
                c_y = cam["params"][2] * scale
                    
            elif model_id == 1: 
                focal_x = cam["params"][0] * scale
                focal_y = cam["params"][1] * scale
                c_x = cam["params"][2] * scale
                c_y = cam["params"][3] * scale

            else:
                raise ValueError(f"other camera_modle is not support....  ID: {model_id}")

            fov_x = 2 * np.arctan(width / (2 * focal_x))
            fov_y = 2 * np.arctan(height / (2 * focal_y))
            '''
            if img_id == next(iter(images_data)): 
                print("--- 縮放正確性檢查 ---")
                print(f"圖片名稱: {img['name']}")
                print(f"縮放比例: {scale}")
                print(f"相機模型 ID: {model_id}")
                print(f"寬高檢查: {width} x {height} (對比 images_4 應為 1237x822)")
                print(f"焦距檢查: fx={focal_x}, fy={focal_y}")
                print(f"FoV 檢查 (弧度): fov_x={fov_x:.4f}, fov_y={fov_y:.4f}")
                print(f"FoV 檢查 (角度): {math.degrees(fov_x):.2f}°")
            '''
            self.poses_dict[img["name"]] = {
                "qvec": img["qvec"],
                "tvec": img["tvec"],
                "fov_x": fov_x,
                "fov_y": fov_y,
                "width": width,
                "height": height
            }
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path).convert("RGB")
        
        img_tensor = TF.to_tensor(img) 
        
        base_name = os.path.splitext(filename)[0]
        mask_filename = base_name + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_filename)
        mask = Image.open(mask_path).convert("L")

        mask_tensor = TF.to_tensor(mask) 
        
        mask_tensor = (mask_tensor > 0.5).float()

        cam_info = self.poses_dict[filename]
        q_tensor = torch.tensor(cam_info["qvec"], dtype=torch.float32)
        R_matrix = quaternion_to_matrix(q_tensor)
        
        camera_dict = {
            "image_name": filename,
            "R": R_matrix,     
            "T": torch.tensor(cam_info["tvec"], dtype=torch.float32),  # 3x1 平移向量
            "FovX": torch.tensor(cam_info["fov_x"], dtype=torch.float32),
            "FovY": torch.tensor(cam_info["fov_y"], dtype=torch.float32),
            "width": cam_info["width"],
            "height": cam_info["height"]
        }
        
        return camera_dict, img_tensor, mask_tensor

    def remove_bad_data(self, bad_photo_names):
        self.image_filenames = [fname for fname in self.image_filenames if fname not in bad_photo_names]
        for bad_name in bad_photo_names:
            if bad_name in self.poses_dict:
                del self.poses_dict[bad_name]


'''
if __name__ == "__main__":

    dataset = BicycleFinetuneDataset(data_dir="/work/goet1019/dataset/mip_nerf360/3dgs_camera_pose/bicycle")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    

    for i, (camera_dict, img_tensor, mask_tensor) in enumerate(dataloader):
        for b in range(img_tensor.shape[0]):
            print(f"--- 正在檢查第 {i*(b+1)+1} 筆資料 ---")
            print(f"圖片名稱: {camera_dict['image_name'][b]}")

        
            print(f"??  影像 Shape: {img_tensor.shape}") # 預期: [1, 3, 822, 1237]
            print(f"? 遮罩 Shape: {mask_tensor.shape}") # 預期: [1, 1, 822, 1237]
            print(f"? 像素範圍: Min={img_tensor.min():.2f}, Max={img_tensor.max():.2f}")
            
            # 檢查相機字典
            print("\n? 相機資訊檢查:")
            print(f" - 圖片名稱: {camera_dict['image_name'][b]}")
            print(f" - R 矩陣 Shape: {camera_dict['R'].shape}")    # 預期: [1, 3, 3]
            print(f" - T 向量 Shape: {camera_dict['T'].shape}")    # 預期: [1, 3]
            print(f" - FoVX: {math.degrees(camera_dict['FovX'][b].item()):.2f}°")
            print(f" - 渲染解析度: {camera_dict['width'][b]} x {camera_dict['height'][b]}")
            
            # 5. 【加分題】視覺化確認 (如果你在有螢幕的環境，或想存檔看一眼)
            # 將 Tensor 轉回圖片存起來，確保沒被壓扁
            save_img = TF.to_pil_image(img_tensor[b])
            save_mask = TF.to_pil_image(mask_tensor[b])
            save_img.save("test_output_img.jpg")
            save_mask.save("test_output_mask.png")
            print("\n? 測試圖檔已儲存為 test_output_img.jpg 與 test_output_mask.png，請確認腳踏車有沒有變形！")
'''