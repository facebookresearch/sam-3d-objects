# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch.nn as nn
import torch

FP16_TYPE = torch.float16

FP16_MODULES = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
]

# If we add sparse modules back in, they are compatible with FP16.
# But for now we don't have them and avoid the dependency on FlashAttention
# Instead using the torch implementation of FlashAttention in SDPA.
try:
    from ..modules import sparse as sp

    FP16_MODULES += [
        sp.SparseConv3d,
        sp.SparseInverseConv3d,
        sp.SparseLinear,
    ]
except ImportError:
    pass

FP16_MODULES = tuple(FP16_MODULES)


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            # p.data = p.data.half()
            p.data = p.data.to(FP16_TYPE)


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.float()


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
