import numpy as np
import torch
from plyfile import PlyData, PlyElement


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