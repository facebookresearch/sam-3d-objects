# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch
import torch.nn.functional as F
import math
import cv2
from scipy.stats import qmc
from easydict import EasyDict as edict
from ..representations.octree import DfsOctree


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
    octree: DfsOctree,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    used_rank=None,
    colors_overwrite=None,
    aux=None,
    halton_sampler=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # lazy import
    if "OctreeTrivecRasterizer" not in globals():
        from diffoctreerast import (
            OctreeVoxelRasterizer,
            OctreeGaussianRasterizer,
            OctreeTrivecRasterizer,
            OctreeDecoupolyRasterizer,
        )

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = edict(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=octree.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        with_distloss=pipe.with_distloss,
        jitter=pipe.jitter,
        debug=pipe.debug,
    )

    positions = octree.get_xyz
    if octree.primitive == "voxel":
        densities = octree.get_density
    elif octree.primitive == "gaussian":
        opacities = octree.get_opacity
    elif octree.primitive == "trivec":
        trivecs = octree.get_trivec
        densities = octree.get_density
        raster_settings.density_shift = octree.density_shift
    elif octree.primitive == "decoupoly":
        decoupolys_V, decoupolys_g = octree.get_decoupoly
        densities = octree.get_density
        raster_settings.density_shift = octree.density_shift
    else:
        raise ValueError(f"Unknown primitive {octree.primitive}")
    depths = octree.get_depth

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None
    shs = octree.get_features
    if octree.primitive in ["voxel", "gaussian"] and colors_overwrite is not None:
        colors_precomp = colors_overwrite
        shs = None

    ret = edict()

    if octree.primitive == "voxel":
        renderer = OctreeVoxelRasterizer(raster_settings=raster_settings)
        rgb, depth, alpha, distloss = renderer(
            positions=positions,
            densities=densities,
            shs=shs,
            colors_precomp=colors_precomp,
            depths=depths,
            aabb=octree.aabb,
            aux=aux,
        )
        ret["rgb"] = rgb
        ret["depth"] = depth
        ret["alpha"] = alpha
        ret["distloss"] = distloss
    elif octree.primitive == "gaussian":
        renderer = OctreeGaussianRasterizer(raster_settings=raster_settings)
        rgb, depth, alpha = renderer(
            positions=positions,
            opacities=opacities,
            shs=shs,
            colors_precomp=colors_precomp,
            depths=depths,
            aabb=octree.aabb,
            aux=aux,
        )
        ret["rgb"] = rgb
        ret["depth"] = depth
        ret["alpha"] = alpha
    elif octree.primitive == "trivec":
        raster_settings.used_rank = (
            used_rank if used_rank is not None else trivecs.shape[1]
        )
        renderer = OctreeTrivecRasterizer(raster_settings=raster_settings)
        rgb, depth, alpha, percent_depth = renderer(
            positions=positions,
            trivecs=trivecs,
            densities=densities,
            shs=shs,
            colors_precomp=colors_precomp,
            colors_overwrite=colors_overwrite,
            depths=depths,
            aabb=octree.aabb,
            aux=aux,
            halton_sampler=halton_sampler,
        )
        ret["percent_depth"] = percent_depth
        ret["rgb"] = rgb
        ret["depth"] = depth
        ret["alpha"] = alpha
    elif octree.primitive == "decoupoly":
        raster_settings.used_rank = (
            used_rank if used_rank is not None else decoupolys_V.shape[1]
        )
        renderer = OctreeDecoupolyRasterizer(raster_settings=raster_settings)
        rgb, depth, alpha = renderer(
            positions=positions,
            decoupolys_V=decoupolys_V,
            decoupolys_g=decoupolys_g,
            densities=densities,
            shs=shs,
            colors_precomp=colors_precomp,
            depths=depths,
            aabb=octree.aabb,
            aux=aux,
        )
        ret["rgb"] = rgb
        ret["depth"] = depth
        ret["alpha"] = alpha

    return ret


class OctreeRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        try:
            import diffoctreerast
        except ImportError:
            print(
                "\033[93m[WARNING] diffoctreerast is not installed. The renderer will be disabled.\033[0m"
            )
            self.unsupported = True
        else:
            self.unsupported = False

        self.pipe = edict(
            {
                "with_distloss": False,
                "with_aux": False,
                "scale_modifier": 1.0,
                "used_rank": None,
                "jitter": False,
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
            }
        )
        self.halton_sampler = qmc.Halton(2, scramble=False)
        self.rendering_options.update(rendering_options)
        self.bg_color = None

    def render(
        self,
        octree: DfsOctree,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        colors_overwrite: torch.Tensor = None,
    ) -> edict:
        """
        Render the octree.

        Args:
            octree (Octree): octree
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color

        Returns:
            edict containing:
                color (torch.Tensor): (3, H, W) rendered color
                depth (torch.Tensor): (H, W) rendered depth
                alpha (torch.Tensor): (H, W) rendered alpha
                distloss (Optional[torch.Tensor]): (H, W) rendered distance loss
                percent_depth (Optional[torch.Tensor]): (H, W) rendered percent depth
                aux (Optional[edict]): auxiliary tensors
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]

        if self.unsupported:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            text_bbox = cv2.getTextSize("Unsupported", cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[
                0
            ]
            origin = (512 - text_bbox[0]) // 2, (512 - text_bbox[1]) // 2
            image = cv2.putText(
                image,
                "Unsupported",
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            return {
                "color": torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
                / 255,
            }

        if self.rendering_options["bg_color"] == "random":
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(
                self.rendering_options["bg_color"], dtype=torch.float32, device="cuda"
            )

        if self.pipe["with_aux"]:
            aux = {
                "grad_color2": torch.zeros(
                    (octree.num_leaf_nodes, 3),
                    dtype=torch.float32,
                    requires_grad=True,
                    device="cuda",
                )
                + 0,
                "contributions": torch.zeros(
                    (octree.num_leaf_nodes, 1),
                    dtype=torch.float32,
                    requires_grad=True,
                    device="cuda",
                )
                + 0,
            }
            for k in aux.keys():
                aux[k].requires_grad_()
                aux[k].retain_grad()
        else:
            aux = None

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
            }
        )

        # Render
        render_ret = render(
            camera_dict,
            octree,
            self.pipe,
            self.bg_color,
            aux=aux,
            colors_overwrite=colors_overwrite,
            scaling_modifier=self.pipe.scale_modifier,
            used_rank=self.pipe.used_rank,
            halton_sampler=self.halton_sampler,
        )

        if ssaa > 1:
            render_ret.rgb = F.interpolate(
                render_ret.rgb[None],
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze()
            render_ret.depth = F.interpolate(
                render_ret.depth[None, None],
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze()
            render_ret.alpha = F.interpolate(
                render_ret.alpha[None, None],
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze()
            if hasattr(render_ret, "percent_depth"):
                render_ret.percent_depth = F.interpolate(
                    render_ret.percent_depth[None, None],
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze()

        ret = edict(
            {
                "color": render_ret.rgb,
                "depth": render_ret.depth,
                "alpha": render_ret.alpha,
            }
        )
        if self.pipe["with_distloss"] and "distloss" in render_ret:
            ret["distloss"] = render_ret.distloss
        if self.pipe["with_aux"]:
            ret["aux"] = aux
        if hasattr(render_ret, "percent_depth"):
            ret["percent_depth"] = render_ret.percent_depth
        return ret
