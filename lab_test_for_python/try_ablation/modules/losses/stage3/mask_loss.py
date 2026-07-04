import torch

def mask_bce_loss(pred_mask, gt_mask, eps=1e-6):
    """
    pred_mask: [1, H, W], value in [0, 1]
    gt_mask:   [1, H, W], value in {0, 1}
    """
    pred_mask = pred_mask.float().clamp(eps, 1.0 - eps)
    gt_mask = gt_mask.float().clamp(0.0, 1.0)

    loss = -(
        gt_mask * torch.log(pred_mask)
        + (1.0 - gt_mask) * torch.log(1.0 - pred_mask)
    )

    return loss.mean()