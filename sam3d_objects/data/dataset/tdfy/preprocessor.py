# Copyright (c) Meta Platforms, Inc. and affiliates.
import warnings
import torch
from loguru import logger
from dataclasses import dataclass
from typing import Callable, Optional
import warnings

from .img_and_mask_transforms import (
    SSIPointmapNormalizer,
)


# Load and process data
@dataclass
class PreProcessor:
    """
    Preprocessor configuration for image, mask, and pointmap transforms.

    Transform application order:
    1. Pointmap normalization (if normalize_pointmap=True)
    2. Joint transforms (img_mask_pointmap_joint_transform or img_mask_joint_transform)
    3. Individual transforms (img_transform, mask_transform, pointmap_transform)

    For backward compatibility, img_mask_joint_transform is preserved. When both
    img_mask_pointmap_joint_transform and img_mask_joint_transform are present,
    img_mask_pointmap_joint_transform takes priority.
    """

    img_transform: Callable = (None,)
    mask_transform: Callable = (None,)
    img_mask_joint_transform: list[Callable] = (None,)
    rgb_img_mask_joint_transform: list[Callable] = (None,)

    # New fields for pointmap support
    pointmap_transform: Callable = (None,)
    img_mask_pointmap_joint_transform: list[Callable] = (None,)
    
    # Pointmap normalization option
    normalize_pointmap: bool = False
    pointmap_normalizer: Optional[Callable] = None
    rgb_pointmap_normalizer: Optional[Callable] = None

    def __post_init__(self):
        if self.pointmap_normalizer is None:
            self.pointmap_normalizer = SSIPointmapNormalizer()
            if self.normalize_pointmap == False:
                warnings.warn("normalize_pointmap is also set to False, which means we will return the moments but not normalize the pointmap. This supports old unnormalized pointmap models, but this is dangerous behavior.", DeprecationWarning, stacklevel=2)

        if self.rgb_pointmap_normalizer is None:
            logger.warning("No rgb pointmap normalizer provided, using scale + shift ")
            self.rgb_pointmap_normalizer = self.pointmap_normalizer


    def _normalize_pointmap(
        self, pointmap: torch.Tensor,
        mask: torch.Tensor,
        pointmap_normalizer: Callable,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ):
        if pointmap is None:
            return pointmap, None, None

        if self.normalize_pointmap == False:
            # old behavior: Pose is normalized to the pointmap center, but pointmap is not
            _, pointmap_scale, pointmap_shift = pointmap_normalizer.normalize(pointmap, mask)
            return pointmap, pointmap_scale, pointmap_shift
        
        if scale is not None or shift is not None:
            return pointmap_normalizer.normalize(pointmap, mask, scale, shift)
            
        return pointmap_normalizer.normalize(pointmap, mask)

    def _process_image_mask_pointmap_mess(
        self, rgb_image, rgb_image_mask, pointmap=None
    ):
        """Extended version that handles pointmaps"""
 
        # Apply pointmap normalization if enabled
        pointmap_for_crop, pointmap_scale, pointmap_shift = self._normalize_pointmap(
            pointmap, rgb_image_mask, self.pointmap_normalizer
        )

        # Apply transforms to the original full rgb image and mask.
        rgb_image, rgb_image_mask = self._preprocess_rgb_image_mask(rgb_image, rgb_image_mask)

        # These two are typically used for getting cropped images of the object
        #   : first apply joint transforms
        processed_rgb_image, processed_mask, processed_pointmap = (
            self._preprocess_image_mask_pointmap(rgb_image, rgb_image_mask, pointmap_for_crop)
        )
        #   : then apply individual transforms on top of the joint transforms
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, self.img_transform
        )
        processed_mask = self._apply_transform(processed_mask, self.mask_transform)
        if processed_pointmap is not None:
            processed_pointmap = self._apply_transform(
                processed_pointmap, self.pointmap_transform
            )

        # This version is typically the full version of the image
        #   : apply individual transforms only
        rgb_image = self._apply_transform(rgb_image, self.img_transform)
        rgb_image_mask = self._apply_transform(rgb_image_mask, self.mask_transform)
        
        rgb_pointmap, rgb_pointmap_scale, rgb_pointmap_shift = self._normalize_pointmap(
            pointmap, rgb_image_mask, self.rgb_pointmap_normalizer, pointmap_scale, pointmap_shift
        )

        if rgb_pointmap is not None:
            rgb_pointmap = self._apply_transform(rgb_pointmap, self.pointmap_transform)

        result = {
            "mask": processed_mask,
            "image": processed_rgb_image,
            "rgb_image": rgb_image,
            "rgb_image_mask": rgb_image_mask,
        }

        # Add pointmap results if available
        if processed_pointmap is not None:
            result.update(
                {
                    "pointmap": processed_pointmap,
                    "rgb_pointmap": rgb_pointmap,
                }
            )
            
        # Add normalization parameters if normalization was applied
        if pointmap_scale is not None and pointmap_shift is not None:
            result.update(
                {
                    "pointmap_scale": pointmap_scale,
                    "pointmap_shift": pointmap_shift,
                    "rgb_pointmap_scale": rgb_pointmap_scale,
                    "rgb_pointmap_shift": rgb_pointmap_shift,
                }
            )

        return result

    def _process_image_and_mask_mess(self, rgb_image, rgb_image_mask):
        """Original method - calls extended version without pointmap"""
        return self._process_image_mask_pointmap_mess(rgb_image, rgb_image_mask, None)

    def _preprocess_rgb_image_mask(self, rgb_image: torch.Tensor, rgb_image_mask: torch.Tensor):
        """Apply joint transforms to rgb_image and rgb_image_mask."""
        if (
            self.rgb_img_mask_joint_transform != (None,)
            and self.rgb_img_mask_joint_transform is not None
        ):
            for trans in self.rgb_img_mask_joint_transform:
                rgb_image, rgb_image_mask = trans(rgb_image, rgb_image_mask)
        return rgb_image, rgb_image_mask

    def _preprocess_image_mask_pointmap(self, rgb_image, mask_image, pointmap=None):
        """Apply joint transforms with priority: triple transforms > dual transforms."""
        # Priority: img_mask_pointmap_joint_transform when pointmap is provided
        if (
            self.img_mask_pointmap_joint_transform != (None,)
            and self.img_mask_pointmap_joint_transform is not None
            and pointmap is not None
        ):
            for trans in self.img_mask_pointmap_joint_transform:
                rgb_image, mask_image, pointmap = trans(
                    rgb_image, mask_image, pointmap=pointmap
                )
            return rgb_image, mask_image, pointmap

        # Fallback: img_mask_joint_transform (existing behavior)
        elif (
            self.img_mask_joint_transform != (None,)
            and self.img_mask_joint_transform is not None
        ):
            for trans in self.img_mask_joint_transform:
                rgb_image, mask_image = trans(rgb_image, mask_image)
            return rgb_image, mask_image, pointmap

        return rgb_image, mask_image, pointmap

    def _preprocess_image_and_mask(self, rgb_image, mask_image):
        """Backward compatibility wrapper - only applies dual transforms"""
        rgb_image, mask_image, _ = self._preprocess_image_mask_pointmap(
            rgb_image, mask_image, None
        )
        return rgb_image, mask_image

    # keep here for backward compatibility
    def _preprocess_image_and_mask_inference(self, rgb_image, mask_image):
        warnings.warn(
            "The _preprocess_image_and_mask_inference is deprecated! Please use _preprocess_image_and_mask",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._preprocess_image_and_mask(rgb_image, mask_image)

    def _apply_transform(self, input: torch.Tensor, transform):
        if input is not None and transform is not None and transform != (None,):
            input = transform(input)

        return input