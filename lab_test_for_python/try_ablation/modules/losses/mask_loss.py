import torch


def mask_l1_loss(pred_mask, gt_mask, eps=1e-6):
    """
    pred_mask: [1, H, W]
    gt_mask:   [1, H, W]
    """

    pred_mask = pred_mask.clamp(0.0, 1.0)
    gt_mask = gt_mask.clamp(0.0, 1.0)

    return torch.abs(pred_mask - gt_mask).mean()


def soft_iou_loss(pred_mask, gt_mask, eps=1e-6):
    """
    Soft IoU loss:
        1 - intersection / union

    pred_mask: [1, H, W]
    gt_mask:   [1, H, W]
    """

    pred_mask = pred_mask.clamp(0.0, 1.0)
    gt_mask = gt_mask.clamp(0.0, 1.0)

    intersection = torch.sum(pred_mask * gt_mask)

    union = (
        torch.sum(pred_mask)
        + torch.sum(gt_mask)
        - intersection
    )

    return 1.0 - intersection / (union + eps)


def mask_loss(
    pred_mask,
    gt_mask,
    lambda_mask_l1=1.0,
    lambda_mask_iou=1.0,
):
    """
    Final mask loss:
        L_mask = lambda_l1 * L1 + lambda_iou * SoftIoU
    """

    loss_l1 = mask_l1_loss(pred_mask, gt_mask)
    loss_iou = soft_iou_loss(pred_mask, gt_mask)

    total = lambda_mask_l1 * loss_l1 + lambda_mask_iou * loss_iou

    return total, {
        "mask_l1": loss_l1.detach(),
        "mask_iou": loss_iou.detach(),
    }