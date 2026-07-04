
import sys

sys.path.append("../notebook")
from inference import Inference, load_image, load_single_mask
import torch
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

tag = "hf"
config_path = f"../checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

image = load_image("../notebook/images/bike/original_image/image0.png")
mask1 = load_single_mask("../notebook/images/bike/samhq_image", index=0)

output = inference(image, mask1, seed=42)

theta = np.radians(90)
c, s = np.cos(theta), np.sin(theta)
R_target = torch.tensor([
    [ c, 0, s],
    [ 0, 1, 0],
    [-s, 0, c]
], dtype=torch.float32, device='cuda')
quats = output['gs'].get_rotation
R_old = quaternion_to_matrix(quats)
R_new = torch.matmul(R_target.unsqueeze(0), R_old)
quats_rotated = matrix_to_quaternion(R_new)

xyz = output['gs'].get_xyz
xyz_rotated = xyz @ R_target.T 
aabb_offset = output['gs'].aabb[None, :3]
aabb_scale = output['gs'].aabb[None, 3:]

t_vec = output['translation'].squeeze() 
t_scale = output['translation_scale'].squeeze()
final_translation = t_vec * t_scale.to(xyz.device)

print(final_translation)
xyz_translated = xyz_rotated + final_translation

xyz_to_save = (xyz_translated - aabb_offset) / aabb_scale
output['gs']._xyz.data.copy_(xyz_to_save)

bias = output['gs'].rots_bias.to(quats_rotated.device)
quats_to_save = quats_rotated - bias[None, :]
output['gs']._rotation.data.copy_(quats_to_save)



output["gs"].save_ply(f"bicycle_for_rotation_and_translation.ply")


print("=== Output Keys ===")
print("=== Output Details ===")
print(output["scale"])
for key, value in output.items():
    if hasattr(value, 'shape'):
        print(f"{key:25} | Type: {type(value).__name__:15} | Shape: {value.shape}")
    
    elif isinstance(value, (int, float, list)):
        print(f"{key:25} | Type: {type(value).__name__:15} | Value: {value}")
    
    else:
        print(f"{key:25} | Type: {type(value).__name__:15} | Info: {value}")
print("======================")






