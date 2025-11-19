# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Union
import torch
import numpy as np
from functools import partial

from PIL import Image


from sam3d_objects.data.dataset.tdfy.preprocessor import PreProcessor
from torchvision.transforms import Compose, Resize, InterpolationMode
from sam3d_objects.data.dataset.tdfy.img_processing import pad_to_square_centered
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    rembg,
    crop_around_mask_with_padding,
)


def get_default_preprocessor():
    preprocessor = PreProcessor()
    img_transform = Compose(
        transforms=[
            partial(pad_to_square_centered),
            Resize(size=518, interpolation=InterpolationMode.BICUBIC),
        ]
    )
    mask_transform = Compose(
        transforms=[
            partial(pad_to_square_centered),
            Resize(size=518, interpolation=0),
        ]
    )
    img_mask_joint_transform = [
        partial(crop_around_mask_with_padding, box_size_factor=1.0, padding_factor=0.1),
        rembg,
    ]
    preprocessor.img_transform = img_transform
    preprocessor.mask_transform = mask_transform
    preprocessor.img_mask_joint_transform = img_mask_joint_transform

    return preprocessor
