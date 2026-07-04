import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_binary_boundary(mask, kernel_size=3):
    """
    Args:
        mask: [1, H, W], binary 0/1

    Returns:
        boundary: [1, H, W], binary boundary map
    """

    if mask.dim() == 3:
        mask_4d = mask.unsqueeze(0)  # [1, 1, H, W]
    else:
        mask_4d = mask

    pad = kernel_size // 2

    dilated = F.max_pool2d(
        mask_4d,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )

    eroded = -F.max_pool2d(
        -mask_4d,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )

    boundary = (dilated - eroded).clamp(0.0, 1.0)

    if mask.dim() == 3:
        boundary = boundary.squeeze(0)

    return boundary


def compute_gt_boundary_dt(gt_mask, normalize=True, eps=1e-6):
    """
    Compute distance transform map from GT mask boundary.

    Args:
        gt_mask: [1, H, W], binary 0/1

    Returns:
        dt_map: [1, H, W], each pixel stores distance to GT boundary
    """

    device = gt_mask.device

    with torch.no_grad():
        gt_mask_cpu = gt_mask.detach().float().cpu()

        gt_boundary = compute_binary_boundary(gt_mask_cpu)
        gt_boundary_np = gt_boundary.squeeze(0).numpy() > 0.5

        # distance_transform_edt computes distance to zero pixels.
        # Here ~gt_boundary_np means each pixel measures distance to boundary.
        dt_np = distance_transform_edt(~gt_boundary_np).astype(np.float32)

        if normalize:
            max_val = dt_np.max()
            if max_val > eps:
                dt_np = dt_np / max_val

        dt_map = torch.from_numpy(dt_np).float().unsqueeze(0).to(device)

    return dt_map


def soft_boundary_from_mask(pred_mask, eps=1e-6):
    """
    Use Sobel gradient to extract differentiable soft boundary.

    Args:
        pred_mask: [1, H, W], soft 0~1

    Returns:
        pred_boundary: [1, H, W]
    """

    device = pred_mask.device
    dtype = pred_mask.dtype

    if pred_mask.dim() == 3:
        x = pred_mask.unsqueeze(0)  # [1, 1, H, W]
    else:
        x = pred_mask

    sobel_x = torch.tensor(
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],
        device=device,
        dtype=dtype,
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]],
        device=device,
        dtype=dtype,
    ).unsqueeze(0)

    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)

    boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)

    if pred_mask.dim() == 3:
        boundary = boundary.squeeze(0)

    return boundary


def dt_loss_stage3(pred_mask, gt_mask, eps=1e-6):
    """
    DT boundary loss.

    Intuition:
        predicted boundary should be close to GT boundary.

    Args:
        pred_mask: [1, H, W], soft 0~1
        gt_mask:   [1, H, W], binary 0/1

    Returns:
        loss
        items
    """

    pred_mask = pred_mask.clamp(0.0, 1.0)
    gt_mask = gt_mask.float().clamp(0.0, 1.0)

    gt_dt = compute_gt_boundary_dt(gt_mask)
    pred_boundary = soft_boundary_from_mask(pred_mask)

    loss = torch.sum(pred_boundary * gt_dt) / (
        torch.sum(pred_boundary) + eps
    )

    return loss, {
        "dt_loss": loss.detach(),
        "pred_boundary_mean": pred_boundary.detach().mean(),
        "gt_dt_mean": gt_dt.detach().mean(),
    }
