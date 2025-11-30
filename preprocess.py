import os
import random
import argparse

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2


def save_sam3d_images(original_image, sam_masks, output_dir):
    """
    Save a single image and its masks in a simple SAM-3D-Objects-style format:

        output_dir/
          image.png
          0.png
          1.png
          ...

    Args:
        original_image: RGB image (H, W, 3) uint8
        sam_masks: list of SAM mask dicts with 'segmentation' and 'bbox' keys
        output_dir: folder to write image + masks into
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure uint8 RGB
    img = original_image
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Save image as PNG (convert RGB -> BGR for OpenCV)
    image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    image_path = os.path.join(output_dir, "image.png")
    cv2.imwrite(image_path, image_bgr)

    # Save each mask as a separate binary PNG: 0.png, 1.png, ...
    for idx, mask in enumerate(sam_masks):
        seg = mask["segmentation"].astype(np.uint8) * 255  # 0/255
        mask_path = os.path.join(output_dir, f"{idx}.png")
        cv2.imwrite(mask_path, seg)

    print(f"  Saved SAM 3D Objects simple format to: {output_dir}")
    print(f"    - Image: image.png")
    mask_msg = (
        f"    - Masks: {len(sam_masks)} individual mask files "
        f"(0.png, 1.png, ...)"
    )
    print(mask_msg)


def save_segmentation_visualizations(
    original_image, original_masks, output_dir
):
    """
    Save visualization images showing SAM segmentations for each mode
    ('default', 's', 'm', 'l') overlaid on the original image.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure uint8 RGB
    img = original_image
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    modes = ["default", "s", "m", "l"]
    alpha = 0.5  # overlay strength

    for mode in modes:
        masks = original_masks.get(mode, [])
        if not masks:
            continue

        vis = img.copy().astype(np.float32)

        # Use a few fixed colors and cycle through them
        colors = [
            np.array([255, 0, 0], dtype=np.float32),    # red
            np.array([0, 255, 0], dtype=np.float32),    # green
            np.array([0, 0, 255], dtype=np.float32),    # blue
            np.array([255, 255, 0], dtype=np.float32),  # yellow
            np.array([255, 0, 255], dtype=np.float32),  # magenta
            np.array([0, 255, 255], dtype=np.float32),  # cyan
        ]

        for idx, m in enumerate(masks):
            seg = m["segmentation"].astype(bool)
            color = colors[idx % len(colors)]
            # alpha blend color onto vis where seg is True
            vis[seg] = (1.0 - alpha) * vis[seg] + alpha * color

        vis = np.clip(vis, 0, 255).astype(np.uint8)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_dir, f"segments_{mode}.png")
        cv2.imwrite(out_path, vis_bgr)
        print(
            f"  Saved segmentation visualization for mode '{mode}' "
            f"to: {out_path}"
        )

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation'] == 0] = np.array(
        [0, 0, 0], dtype=np.uint8
    )
    x, y, w, h = np.int32(mask['bbox'])
    seg_img = image[y:y + h, x:x + w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep:
            result_keep.append(m)
    return result_keep

def mask_nms(
    masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs
):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based
    on their scores.

    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape
            (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected
            indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros(
        (num_masks,) * 2, dtype=torch.float, device=masks.device
    )
    inner_iou_matrix = torch.zeros(
        (num_masks,) * 2, dtype=torch.float, device=masks.device
    )
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(
                torch.logical_and(masks_ord[i], masks_ord[j]),
                dtype=torch.float
            )
            union = torch.sum(
                torch.logical_or(masks_ord[i], masks_ord[j]),
                dtype=torch.float
            )
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal
            # relationship
            if (intersection / masks_area[i] < 0.5 and
                    intersection / masks_area[j] >= 0.85):
                inner_iou = (
                    1 - (intersection / masks_area[j]) *
                    (intersection / masks_area[i])
                )
                inner_iou_matrix[i, j] = inner_iou
            if (intersection / masks_area[i] >= 0.85 and
                    intersection / masks_area[j] < 0.5):
                inner_iou = (
                    1 - (intersection / masks_area[j]) *
                    (intersection / masks_area[i])
                )
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold,
    # the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    # Extract max_large_segments if provided
    max_large_segments = kwargs.pop('max_large_segments', None)
    
    for idx, masks_lvl in enumerate(args):
        seg_pred = torch.from_numpy(
            np.stack([m['segmentation'] for m in masks_lvl], axis=0)
        )
        iou_pred = torch.from_numpy(
            np.stack([m['predicted_iou'] for m in masks_lvl], axis=0)
        )
        stability = torch.from_numpy(
            np.stack([m['stability_score'] for m in masks_lvl], axis=0)
        )

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)
        
        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    image = cv2.cvtColor(
        image[0].permute(1, 2, 0).numpy().astype(np.uint8),
        cv2.COLOR_BGR2RGB
    )
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # Store original masks before postprocessing for SAM 3D Objects
    original_masks = {
        'default': masks_default.copy(),
        's': masks_s.copy() if len(masks_s) > 0 else [],
        'm': masks_m.copy() if len(masks_m) > 0 else [],
        'l': masks_l.copy() if len(masks_l) > 0 else []
    }
    # pre-compute postprocess (NMS etc.);
    # do NOT limit count of large masks here
    masks_default, masks_s, masks_m, masks_l = masks_update(
        masks_default, masks_s, masks_m, masks_l,
        iou_thr=0.8, score_thr=0.7, inner_thr=0.5
    )
    
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0)  # b,H,W,3
        seg_imgs = (
            torch.from_numpy(seg_imgs.astype("float32"))
            .permute(0, 3, 1, 2) / 255.0
        ).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    (
        seg_images['default'],
        seg_maps['default']
    ) = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(
            masks_s, image
        )
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(
            masks_m, image
        )
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(
            masks_l, image
        )
    
    # 0:default 1:s 2:m 3:l
    # Return both processed seg_maps and original masks
    # for SAM 3D Objects
    return seg_images, seg_maps, original_masks, image

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def process_single_image(
    image_path, output_dir, mask_generator
):
    """
    Process a single image and save the results in separate
    subdirectories for each mode (s, m, l).
    
    Output structure:
        output_dir/
        s_result.png
        m_result.png
        l_result.png
          s/
            image.png
            0.png
            1.png
            ...
          m/
            image.png
            0.png
            1.png
            ...
          l/
            image.png
            0.png
            1.png
            ...
    
    Args:
        image_path: Path to the image file
        output_dir: Output directory for results
        mask_generator: Initialized SAM mask generator
    """
    # Load image (BGR, H x W x 3)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(
            f"[WARN] Could not read image at '{image_path}'; skipping"
        )
        return False

    # Build single-image batch tensor for SAM encoder
    image_tensor = torch.from_numpy(image_bgr)  # H, W, 3
    images = image_tensor.permute(2, 0, 1)[None, ...]  # 1, 3, H, W

    print(f"[INFO] Processing image: {image_path}")
    (
        seg_images, seg_maps, original_masks, original_image
    ) = sam_encoder(images)

    # Save visualization images showing segmentation overview
    # for each mode
    save_segmentation_visualizations(original_image, original_masks, output_dir)

    # Save results for each mode (s, m, l) in separate subdirectories
    modes_to_save = ['s', 'm', 'l']
    saved_any = False
    
    for mode in modes_to_save:
        masks = original_masks.get(mode, [])
        if not masks:
            print(
                f"[WARN] No '{mode}' masks found; skipping"
            )
            continue
        
        # Create subdirectory for this mode
        mode_output_dir = os.path.join(output_dir, mode)
        
        # Save image and masks for this mode
        print(
            f"[INFO] Saving {len(masks)} '{mode}' masks to "
            f"'{mode_output_dir}'"
        )
        save_sam3d_images(original_image, masks, mode_output_dir)
        saved_any = True

    if not saved_any:
        print(
        "[ERROR] No masks found for any mode (s, m, l); "
    )
        return False
    
    return True




def preprocess(input_dir, output_dir, sam_ckpt_path):
    seed_num = 42
    seed_everything(seed_num)
    torch.set_default_dtype(torch.float32)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input directory '{input_dir}' does not exist"
        )

    # Get the directory name to use for output
    input_dir_name = os.path.basename(os.path.normpath(input_dir))
    final_output_dir = os.path.join(
        output_dir, input_dir_name
    )
    
    # Initialize SAM model and mask generator
    print("[INFO] Initializing SAM model...")
    sam = sam_model_registry["vit_h"](
        checkpoint=sam_ckpt_path
    ).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    # Expose mask_generator globally for sam_encoder
    globals()["mask_generator"] = mask_generator

    # Find image.png in the input directory
    image_path = os.path.join(input_dir, 'image.png')
    
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"image.png not found in '{input_dir}'"
        )
    
    # Process the image
    print(f"[INFO] Processing image: {image_path}")
    print(f"[INFO] Saving results to: {final_output_dir}")
    process_single_image(image_path, final_output_dir, mask_generator)
    
    print(f"\n[INFO] Finished processing directory '{input_dir}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to directory containing images to process')
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help=(
            'Output folder: will contain processed results with '
            'same directory name'
        )
    )
    parser.add_argument(
        '--sam_ckpt_path', type=str,
        default="ckpts/sam_vit_h_4b8939.pth"
    )
    args = parser.parse_args()
    preprocess(args.input_dir, args.output_dir, args.sam_ckpt_path)
