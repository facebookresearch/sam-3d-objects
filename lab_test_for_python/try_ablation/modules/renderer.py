import math
import torch
import gsplat

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

def render_zi(
    means,
    scales,
    quats,
    opacities,
    membership_logit,
    camera_dict,
):
    """
    Render per-Gaussian membership probability z_i into a 2D soft mask.

    membership_logit: [N], learnable parameter
    return:
        pred_mask: [ 1, H, W], value in [0, 1]
    """

    membership_prob = torch.sigmoid(membership_logit)  # [N]

    # z_i as grayscale RGB color
    membership_colors = membership_prob[:, None].repeat(1, 3)  # [N, 3]

    rendered = render_standalone(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        colors=membership_colors,
        camera_dict=camera_dict,
    )  # [H, W, 3]

    pred_mask = rendered[..., 0]  # [H, W]

    pred_mask = pred_mask[None, :, :]  # [1, H, W]
    pred_mask = pred_mask.clamp(1e-6, 1.0 - 1e-6)

    return pred_mask, membership_prob
