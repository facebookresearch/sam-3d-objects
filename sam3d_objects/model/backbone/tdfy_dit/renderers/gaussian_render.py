# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from easydict import EasyDict as edict
import numpy as np
from ..representations.gaussian import Gaussian
from .sh_utils import eval_sh
import torch.nn.functional as F
from easydict import EasyDict as edict
import warnings

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizer,
        GaussianRasterizationSettings,
    )
except ImportError:
    warnings.warn(
        "'diff_gaussian_rasterization' module cannot be imported, backend 'inria' won't be available",
        ImportWarning,
    )

from gsplat import rasterization


def intrinsics_to_projection(
    intrinsics: torch.Tensor,
    near: float,
    far: float,
) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


def render(
    viewpoint_camera,
    pc: Gaussian,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    backend="inria",
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Backend-specific rasterization setup and execution
    if backend == "inria":
        kernel_size = pipe.kernel_size
        subpixel_offset = torch.zeros(
            (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
            dtype=torch.float32,
            device="cuda",
        )

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=kernel_size,
            subpixel_offset=subpixel_offset,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
    elif backend == "gsplat":
        """
        See reference code to convert from gsplat to inria:
        https://github.com/nerfstudio-project/gsplat/blob/2323de5905d5e90e035f792fe65bad0fedd413e7/gsplat/rendering.py#L1108
        """
        # Unnormalize the intrinsics matrix to get pixel coordinates
        Ks = viewpoint_camera.intrinsics.clone().unsqueeze(0)  # Add batch dimension
        Ks[0, 0, 0] *= viewpoint_camera.image_width  # fx
        Ks[0, 1, 1] *= viewpoint_camera.image_height  # fy
        Ks[0, 0, 2] *= viewpoint_camera.image_width  # cx
        Ks[0, 1, 2] *= viewpoint_camera.image_height  # cy

        # For gsplat, when using SH coefficients, pass them as colors with sh_degree set
        gsplat_colors = colors_precomp if colors_precomp is not None else shs
        gsplat_sh_degree = pc.active_sh_degree if shs is not None else None

        render_colors, render_alphas, meta = rasterization(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=gsplat_colors,
            sh_degree=gsplat_sh_degree,
            viewmats=viewpoint_camera.world_view_transform.T.contiguous().unsqueeze(0),
            Ks=Ks,
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            backgrounds=bg_color,
        )
        rendered_image = render_colors.squeeze(0).permute(
            2, 0, 1
        )  # Convert to (C, H, W)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return edict(
        {
            "render": rendered_image,
        }
    )


class GaussianRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.pipe = edict(
            {
                "kernel_size": 0.1,
                "convert_SHs_python": False,
                "compute_cov3D_python": False,
                "scale_modifier": 1.0,
                "debug": False,
            }
        )

        self.rendering_options = edict(
            {
                "resolution": None,
                "near": None,
                "far": None,
                "ssaa": 1,
                "bg_color": "random",
                "backend": "inria",
            }
        )
        self.rendering_options.update(rendering_options)
        self.bg_color = None

    def render(
        self,
        gausssian: Gaussian,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        colors_overwrite: torch.Tensor = None,
    ) -> edict:
        """
        Render the gausssian.

        Args:
            gaussian : gaussianmodule
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color

        Returns:
            edict containing:
                color (torch.Tensor): (3, H, W) rendered color image
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]

        if self.rendering_options["bg_color"] == "random":
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(
                self.rendering_options["bg_color"], dtype=torch.float32, device="cuda"
            )

        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)
        fovy = 2 * torch.atan(0.5 / focaly)

        camera_dict = edict(
            {
                "image_height": resolution * ssaa,
                "image_width": resolution * ssaa,
                "FoVx": fovx,
                "FoVy": fovy,
                "znear": near,
                "zfar": far,
                "world_view_transform": view.T.contiguous(),
                "projection_matrix": perspective.T.contiguous(),
                "full_proj_transform": (perspective @ view).T.contiguous(),
                "camera_center": camera,
                "intrinsics": intrinsics,
            }
        )

        # Render
        render_ret = render(
            camera_dict,
            gausssian,
            self.pipe,
            self.bg_color,
            override_color=colors_overwrite,
            scaling_modifier=self.pipe.scale_modifier,
            backend=self.rendering_options["backend"],
        )

        if ssaa > 1:
            render_ret.render = F.interpolate(
                render_ret.render[None],
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze()

        ret = edict({"color": render_ret["render"]})
        return ret
