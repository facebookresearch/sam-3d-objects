import torch
from pytorch_msssim import ssim


def masked_l1_loss(rendered_image, gt_image, mask, eps=1e-6):
    if mask.shape[0] == 1 and rendered_image.shape[0] == 3:
        mask = mask.expand_as(rendered_image)

    diff = torch.abs(rendered_image - gt_image) * mask

    return diff.sum() / (mask.sum() + eps)


def get_ssim_loss(rendered_image, gt_image):
    img1 = rendered_image.unsqueeze(0)
    img2 = gt_image.unsqueeze(0)

    ssim_value = ssim(
        img1,
        img2,
        data_range=1.0,
        size_average=True,
    )

    return 1.0 - ssim_value


def photo_loss(
    rendered_image,
    gt_image,
    gt_mask,
    lambda_dssim=0.2,
    disable_ssim=False,
):
    loss_l1 = masked_l1_loss(
        rendered_image,
        gt_image,
        gt_mask,
    )

    if disable_ssim or lambda_dssim <= 0:
        return loss_l1, {
            "l1": loss_l1.detach(),
            "ssim": torch.tensor(0.0, device=rendered_image.device),
        }

    if gt_mask.shape[0] == 1 and rendered_image.shape[0] == 3:
        mask_rgb = gt_mask.expand_as(rendered_image)
    else:
        mask_rgb = gt_mask

    masked_rendered = rendered_image * mask_rgb
    masked_gt = gt_image * mask_rgb

    loss_ssim = get_ssim_loss(masked_rendered, masked_gt)

    loss_photo = (1.0 - lambda_dssim) * loss_l1 + lambda_dssim * loss_ssim

    return loss_photo, {
        "l1": loss_l1.detach(),
        "ssim": loss_ssim.detach(),
    }