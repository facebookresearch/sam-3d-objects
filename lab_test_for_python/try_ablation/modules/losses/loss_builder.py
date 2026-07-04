import torch

from modules.losses.photo_loss import photo_loss
from modules.losses.mask_loss import mask_loss
from modules.arap import compute_arap_loss
from modules.losses.dt_loss import dt_boundary_loss
from modules.losses.stage3.mask_loss import mask_bce_loss
from modules.losses.stage3.dt_loss import dt_loss_stage3
from modules.losses.stage3.sam3d_loss import sam3d_prior_loss
from modules.losses.stage3.tv_loss import tv_smoothness_loss
def compute_dynamic_weight(
    loss_value,
    target_ratio,
    eps=1e-8,
    min_weight=1e-4,
    max_weight=100.0,
):
    weight = target_ratio / (loss_value.detach() + eps)
    weight = torch.clamp(weight, min=min_weight, max=max_weight)
    return weight


def compute_total_losses(
    cfg,
    render_out,
    gt_image,
    gt_mask,
    control_coord,
    control_T,
    control_q,
    arap_edge_index,
    arap_edge_weight,
    pred_mask,
):
    device = render_out.device

    total_loss = torch.tensor(0.0, device=device)

    loss_photo = torch.tensor(0.0, device=device)
    loss_l1 = torch.tensor(0.0, device=device)
    loss_ssim = torch.tensor(0.0, device=device)

    loss_arap = torch.tensor(0.0, device=device)
    
    loss_dt = torch.tensor(0.0, device=device)

    lambda_photo_used = torch.tensor(0.0, device=device)
    lambda_arap_used = torch.tensor(0.0, device=device)
    lambda_loss_dt_used = torch.tensor(0.0, device=device)



    # -------------------------
    # Photo loss
    # -------------------------
    if cfg.loss.use_photo:
        loss_photo, photo_items = photo_loss(
            rendered_image=render_out,
            gt_image=gt_image,
            gt_mask=gt_mask,
            lambda_dssim=cfg.loss.lambda_dssim,
            disable_ssim=cfg.loss.disable_ssim,
        )

        loss_l1 = photo_items["l1"]
        loss_ssim = photo_items["ssim"]

    # -------------------------
    # ARAP loss
    # -------------------------
    if cfg.loss.use_arap:
        loss_arap = compute_arap_loss(
            control_coord=control_coord,
            control_T=control_T,
            control_q=control_q,
            edge_index=arap_edge_index,
            edge_weight=arap_edge_weight,
            arap_type=cfg.loss.arap_type,
            normalized=cfg.loss.normalized_arap,
        )

    # -------------------------
    # Mask loss
    # -------------------------
    if cfg.loss.use_mask:
        loss_mask, mask_items = mask_loss(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            lambda_mask_l1=cfg.loss.lambda_mask_l1,
            lambda_mask_iou=cfg.loss.lambda_mask_iou,
        )

    # -------------------------
    # dt loss
    # -------------------------
    if cfg.loss.use_dt:
        loss_dt, _ = dt_loss_stage3(pred_mask=pred_mask, gt_mask=gt_mask)


    # -------------------------
    # Static or adaptive weighting
    # -------------------------
    use_balance = hasattr(cfg, "loss_balance") and cfg.loss_balance.enabled

    if use_balance:
        if cfg.loss.use_photo:
            lambda_photo_used = compute_dynamic_weight(
                loss_photo,
                target_ratio=cfg.loss_balance.photo_target,
                eps=cfg.loss_balance.eps,
                min_weight=cfg.loss_balance.min_weight,
                max_weight=cfg.loss_balance.max_weight,
            )

        if cfg.loss.use_arap:
            lambda_arap_used = compute_dynamic_weight(
                loss_arap,
                target_ratio=cfg.loss_balance.arap_target,
                eps=cfg.loss_balance.eps,
                min_weight=cfg.loss_balance.min_weight,
                max_weight=cfg.loss_balance.max_weight,
            )

        if cfg.loss.use_mask:
            lambda_mask_used = compute_dynamic_weight(
                loss_mask,
                target_ratio=cfg.loss_balance.mask_target,
                eps=cfg.loss_balance.eps,
                min_weight=cfg.loss_balance.min_weight,
                max_weight=cfg.loss_balance.max_weight,
            )
        if cfg.loss.use_dt:
            lambda_loss_dt_used = compute_dynamic_weight(
                loss_dt,
                target_ratio=cfg.loss_balance.dt_target,
                eps=cfg.loss_balance.eps,
                min_weight=cfg.loss_balance.min_weight,
                max_weight=cfg.loss_balance.max_weight,
            )

    else:
        lambda_photo_used = torch.tensor(cfg.loss.lambda_photo, device=device)
        lambda_arap_used = torch.tensor(cfg.loss.lambda_arap, device=device)
        lambda_mask_used = torch.tensor(cfg.loss.lambda_mask, device=device)
        lambda_loss_dt_used = torch.tensor(cfg.loss.lambda_dt, device=device)
    # -------------------------
    # Total loss
    # -------------------------
    weighted_photo_loss = lambda_photo_used * loss_photo
    weighted_arap_loss = lambda_arap_used * loss_arap
    weighted_mask_loss = lambda_mask_used * loss_mask
    weighted_dt_loss = lambda_loss_dt_used * loss_dt

    if cfg.loss.use_photo:
        total_loss = total_loss + weighted_photo_loss

    if cfg.loss.use_arap:
        total_loss = total_loss + weighted_arap_loss

    if cfg.loss.use_mask:
        total_loss = total_loss + weighted_mask_loss

    if cfg.loss.use_dt:
        total_loss = total_loss + weighted_dt_loss

    total_detached = total_loss.detach() + 1e-8

    photo_ratio = weighted_photo_loss.detach() / total_detached
    arap_ratio = weighted_arap_loss.detach() / total_detached
    mask_ratio = weighted_mask_loss.detach() / total_detached
    dt_ratio = weighted_dt_loss.detach() / total_detached
    return {
        "total_loss": total_loss,

        "photo_loss": loss_photo,
        "l1_loss": loss_l1,
        "ssim_loss": loss_ssim,

        "arap_loss": loss_arap,
        "dt_loss": loss_dt,
        
        "mask_loss": loss_mask,
        "weighted_photo_loss": weighted_photo_loss,
        "weighted_arap_loss": weighted_arap_loss,
        "weighted_mask_loss": weighted_mask_loss,
        "weighted_dt_loss": weighted_dt_loss,

        "lambda_photo_used": lambda_photo_used.detach(),
        "lambda_arap_used": lambda_arap_used.detach(),
        "lambda_mask_used": lambda_mask_used.detach(),
        "lambda_dt_used": lambda_loss_dt_used.detach(),

        "photo_ratio": photo_ratio.detach(),
        "arap_ratio": arap_ratio.detach(),
        "mask_ratio": mask_ratio.detach(),
        "dt_ratio": dt_ratio.detach(),
    }

def compute_total_losses_stage3(
    cfg,
    render_out,
    gt_image,
    gt_mask,
    membership_prob=None,
    sam3d_target=None,
    tv_edge_index=None,
    tv_edge_weight=None,
    step=0,
):
    device = render_out.device
    total_loss = torch.tensor(0.0, device=device)
    loss_mask = torch.tensor(0.0, device=device)
    loss_dt = torch.tensor(0.0, device=device)
    loss_sam3d = torch.tensor(0.0, device=device)
    loss_tv = torch.tensor(0.0, device=device)


    weighted_mask_loss = torch.tensor(0.0, device=device)
    weighted_dt_loss = torch.tensor(0.0, device=device)
    weighted_sam3d_loss = torch.tensor(0.0, device=device)
    weighted_tv_loss = torch.tensor(0.0, device=device)
    # -------------------------
    # Mask loss
    # -------------------------
    if cfg.stage3_finetune.loss.use_mask:
        loss_mask = mask_bce_loss(
            pred_mask=render_out,
            gt_mask=gt_mask,
        )
        lambda_mask_used = torch.tensor(
            cfg.stage3_finetune.loss.lambda_mask,
            device=device
        )
        weighted_mask_loss = lambda_mask_used * loss_mask
        total_loss = total_loss + weighted_mask_loss


    if cfg.stage3_finetune.loss.use_dt:
        loss_dt,_ = dt_loss_stage3(
            pred_mask=render_out,
            gt_mask=gt_mask,
        )

        lambda_dt_used = torch.tensor(
            cfg.stage3_finetune.loss.lambda_dt,
            device=device
        )

        weighted_dt_loss = lambda_dt_used * loss_dt
        total_loss = total_loss + weighted_dt_loss
    if cfg.stage3_finetune.loss.use_sam3d:
        assert membership_prob is not None, \
            "membership_prob is required when use_sam3d=True"
        assert sam3d_target is not None, \
            "sam3d_target is required when use_sam3d=True"

        loss_sam3d = sam3d_prior_loss(
            membership_prob=membership_prob,
            sam3d_target=sam3d_target,
            neg_weight=cfg.stage3_finetune.loss.sam3d_neg_weight,
        )

        lambda_sam3d_used = torch.tensor(
            cfg.stage3_finetune.loss.lambda_sam3d,
            device=device,
        )

        weighted_sam3d_loss = lambda_sam3d_used * loss_sam3d
        total_loss = total_loss + weighted_sam3d_loss

    
    if cfg.stage3_finetune.loss.use_tv:
        tv_start_iter = cfg.stage3_finetune.loss.tv_start_iter

        if step >= tv_start_iter:
            assert membership_prob is not None, \
                "membership_prob is required when use_tv=True"
            assert tv_edge_index is not None, \
                "tv_edge_index is required when use_tv=True"

            loss_tv = tv_smoothness_loss(
                membership_prob=membership_prob,
                edge_index=tv_edge_index,
                edge_weight=tv_edge_weight,
            )

            lambda_tv_used = torch.tensor(
                cfg.stage3_finetune.loss.lambda_tv,
                device=device,
            )

            weighted_tv_loss = lambda_tv_used * loss_tv
            total_loss = total_loss + weighted_tv_loss

    return {
        "total_loss": total_loss,

        # -------------------------
        # Mask loss
        # -------------------------
        "mask_loss": loss_mask,
        "weighted_mask_loss": weighted_mask_loss,
        # -------------------------
        # DT loss
        # -------------------------
        "dt_loss": loss_dt,
        "weighted_dt_loss": weighted_dt_loss,
        # -------------------------
        # SAM3D loss
        # -------------------------
        "sam3d_loss": loss_sam3d,
        "weighted_sam3d_loss": weighted_sam3d_loss,
        # -------------------------
        # TV loss
        # -------------------------
        "tv_loss": loss_tv,
        "weighted_tv_loss": weighted_tv_loss,
    }

