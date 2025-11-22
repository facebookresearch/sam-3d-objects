"""
Load multi-view data from specified path
Supports two data structures:
1. mask_prompt=None: All images and masks in the same directory, naming format: xxxx.png and xxxx_mask.png
2. mask_prompt!=None: Images in input_path/images/, masks in input_path/{mask_prompt}/
"""
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from loguru import logger


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array"""
    img = Image.open(path)
    return np.array(img).astype(np.uint8)


def load_mask_from_rgba(path: Path) -> np.ndarray:
    """
    Load mask from RGBA image (extract from alpha channel)
    
    Args:
        path: RGBA image file path
        
    Returns:
        mask: Binary mask, shape (H, W), bool format
    """
    img = Image.open(path)
    img_array = np.array(img)
    
    if img.mode == 'RGBA' and img_array.ndim == 3 and img_array.shape[2] >= 4:
        mask = img_array[..., 3] > 0
    elif img.mode == 'RGB':
        logger.warning(f"Mask file {path} is RGB format, not RGBA. Using all pixels as mask.")
        mask = np.ones((img_array.shape[0], img_array.shape[1]), dtype=bool)
    else:
        logger.warning(f"Unexpected image mode {img.mode} for mask file {path}")
        mask = np.ones((img_array.shape[0], img_array.shape[1]), dtype=bool)
    
    return mask


def load_images_and_masks(
    images_and_masks_dir: Path,
    image_names: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load multi-view data from images_and_masks folder
    
    Data structure:
    images_and_masks/
        ├── 1.png (or image1.png, view_a.png, etc.)
        ├── 1_mask.png (or image1_mask.png, view_a_mask.png)
        ├── 2.png
        ├── 2_mask.png
        └── ...
    
    Args:
        images_and_masks_dir: Path to images_and_masks folder
        image_names: List of image names (without extension), e.g., ["image1", "view_a"] or ["1", "2"],
                     if None then auto-detect all
        
    Returns:
        images: List of images (numpy arrays)
        masks: List of masks (numpy arrays, bool format)
    """
    if not images_and_masks_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {images_and_masks_dir}")
    
    if not images_and_masks_dir.is_dir():
        raise ValueError(f"Path is not a directory: {images_and_masks_dir}")
    
    if image_names is None:
        image_files = sorted(images_and_masks_dir.glob("*.png")) + sorted(images_and_masks_dir.glob("*.jpg"))
        image_files = [f for f in image_files if "_mask" not in f.name]
        image_names = [f.stem for f in image_files]
        logger.info(f"Auto-detected {len(image_names)} images: {image_names}")
    
    images = []
    masks = []
    
    for image_name in image_names:
        image_candidates = [
            images_and_masks_dir / f"{image_name}.png",
            images_and_masks_dir / f"{image_name}.jpg",
        ]
        
        mask_candidates = [
            images_and_masks_dir / f"{image_name}_mask.png",
            images_and_masks_dir / f"{image_name}_mask.jpg",
        ]
        
        image_path = None
        for candidate in image_candidates:
            if candidate.exists():
                image_path = candidate
                break
        
        mask_path = None
        for candidate in mask_candidates:
            if candidate.exists():
                mask_path = candidate
                break
        
        if image_path is None:
            logger.warning(f"Image file not found for '{image_name}', skipping")
            continue
        
        if mask_path is None:
            logger.warning(f"Mask file not found for '{image_name}', skipping")
            continue
        
        try:
            image = load_image(image_path)
            mask = load_mask_from_rgba(mask_path)
            
            images.append(image)
            masks.append(mask)
            
            logger.info(f"Loaded '{image_name}': image={image.shape}, mask={mask.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load '{image_name}': {e}")
            continue
    
    if len(images) == 0:
        raise ValueError(f"No valid images and masks found in {images_and_masks_dir}")
    
    logger.info(f"Successfully loaded {len(images)} images")
    return images, masks


def load_from_segmentation_structure(
    segmentation_base_dir: Path,
    prompt: Optional[str] = None,
    view_indices: Optional[List[int]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load data from complete segmentation structure (kept for backward compatibility)
    
    Structure:
    visualization/
        └── {folder_name}_segmentation/
            └── {prompt}/
                └── images_and_masks/
    
    Args:
        segmentation_base_dir: Segmentation base directory (e.g., visualization/joy_segmentation)
        prompt: Prompt subfolder name (e.g., "stuffed_toy"), if None then auto-detect
        view_indices: List of view indices, if None then auto-detect all
        
    Returns:
        images: List of images
        masks: List of masks
    """
    if prompt:
        segmentation_dir = segmentation_base_dir / prompt
    else:
        prompt_dirs = [d for d in segmentation_base_dir.iterdir() 
                      if d.is_dir() and d.name != "all_masks"]
        if len(prompt_dirs) == 0:
            raise ValueError(f"No prompt subdirectories found in {segmentation_base_dir}")
        elif len(prompt_dirs) == 1:
            segmentation_dir = prompt_dirs[0]
            logger.info(f"Auto-detected prompt directory: {segmentation_dir.name}")
        else:
            raise ValueError(
                f"Multiple prompt directories found in {segmentation_base_dir}. "
                f"Please specify --prompt. Found: {[d.name for d in prompt_dirs]}"
            )
    
    images_and_masks_dir = segmentation_dir / "images_and_masks"
    
    if not images_and_masks_dir.exists():
        raise FileNotFoundError(
            f"images_and_masks directory not found: {images_and_masks_dir}"
        )
    
    if view_indices is not None and len(view_indices) > 0 and isinstance(view_indices[0], int):
        image_names = [str(idx) for idx in view_indices]
    else:
        image_names = view_indices
    return load_images_and_masks(images_and_masks_dir, image_names=image_names)


def load_images_and_masks_from_path(
    input_path: Path,
    mask_prompt: Optional[str] = None,
    image_names: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load multi-view data from specified path (supports two data structures)
    
    Data structure 1 (mask_prompt=None):
    input_path/
        ├── 1.png
        ├── 1_mask.png
        ├── 2.png
        ├── 2_mask.png
        └── ...
    
    Data structure 2 (mask_prompt!=None, e.g., mask_prompt="stuffed_toy"):
    input_path/
        ├── images/
        │   ├── 1.png
        │   ├── 2.png
        │   └── ...
        └── stuffed_toy/  (or {mask_prompt}/)
            ├── 1.png (or 1_mask.png)
            ├── 2.png (or 2_mask.png)
            └── ...
    
    Args:
        input_path: Input path
        mask_prompt: Mask folder name, if None then images and masks are in the same directory
        image_names: List of image names (without extension), e.g., ["image1", "view_a"] or ["1", "2"],
                     if None then auto-detect all
        
    Returns:
        images: List of images
        masks: List of masks
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_path}")
    
    if mask_prompt is None:
        logger.info(f"Loading from single directory: {input_path}")
        return load_images_and_masks(input_path, image_names=image_names)
    else:
        images_dir = input_path / "images"
        masks_dir = input_path / mask_prompt
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {masks_dir}")
        
        logger.info(f"Loading images from: {images_dir}")
        logger.info(f"Loading masks from: {masks_dir}")
        
        if image_names is None:
            image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
            image_names = [f.stem for f in image_files]
            logger.info(f"Auto-detected {len(image_names)} images: {image_names}")
        
        images = []
        masks = []
        
        for image_name in image_names:
            image_candidates = [
                images_dir / f"{image_name}.png",
                images_dir / f"{image_name}.jpg",
            ]
            
            mask_candidates = [
                masks_dir / f"{image_name}.png",
                masks_dir / f"{image_name}_mask.png",
                masks_dir / f"{image_name}.jpg",
                masks_dir / f"{image_name}_mask.jpg",
            ]
            
            image_path = None
            for candidate in image_candidates:
                if candidate.exists():
                    image_path = candidate
                    break
            
            mask_path = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if image_path is None:
                logger.warning(f"Image file not found for '{image_name}', skipping")
                continue
            
            if mask_path is None:
                logger.warning(f"Mask file not found for '{image_name}', skipping")
                continue
            
            try:
                image = load_image(image_path)
                mask = load_mask_from_rgba(mask_path)
                
                images.append(image)
                masks.append(mask)
                
                logger.info(f"Loaded '{image_name}': image={image.shape}, mask={mask.shape}")
                
            except Exception as e:
                logger.error(f"Failed to load '{image_name}': {e}")
                continue
        
        if len(images) == 0:
            raise ValueError(f"No valid images and masks found in {input_path}")
        
        logger.info(f"Successfully loaded {len(images)} images")
        return images, masks

