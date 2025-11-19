# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from typing import Optional

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import quaternion_to_matrix

from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from sam3d_objects.utils.visualization.plotly.plot_scene import plot_tdfy_scene
from sam3d_objects.utils.visualization.image_mesh import (
    mesh_from_pointmap,
    create_textured_mesh,
)

from sam3d_objects.utils.visualization.plotly.plot_scene import NO_BACKGROUND, default_axisargs
from sam3d_objects.utils.visualization.plotly.save_scene import make_video as make_scene_video
import seaborn as sns
import copy


class SceneVisualizer:
    make_video_from_fig = make_scene_video

    @staticmethod
    def plot_scene(
        points_local: torch.Tensor,
        instance_quaternions_l2c: torch.Tensor,
        instance_positions_l2c: torch.Tensor,
        instance_scales_l2c: torch.Tensor,
        pointmap: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        title: str = "Tdfy Scene",
        height: int = 1000,
        show_pointmap_as_mesh: bool = True,
        clip_pointmap_colors_for_vis: bool = False,
        filter_pointmap_edges: bool = True,
    ):
        cam = SceneVisualizer.camera()

        object_points = SceneVisualizer.object_pointcloud(
            points_local=points_local.unsqueeze(0),
            quat_l2c=instance_quaternions_l2c,
            trans_l2c=instance_positions_l2c,
            scale_l2c=instance_scales_l2c,
            # colors=torch.ones_like(sample["instance_points_local"]) * torch.tensor([1, 0, 0]),
        )

        pointmap_struct_dict = SceneVisualizer._create_pointmap_structure(
            pointmap=pointmap,
            image=image,
            show_pointmap_as_mesh=show_pointmap_as_mesh,
            clip_pointmap_colors_for_vis=clip_pointmap_colors_for_vis,
            filter_pointmap_edges=filter_pointmap_edges,
        )
        return plot_tdfy_scene(
            {
                title: {
                    "camera": cam,
                    "object_points": object_points,
                    **pointmap_struct_dict,
                }
            },
            height=height,
        )

    @staticmethod
    def plot_multi_objects(
        pose_targets,
        mask_names=None,
        pointmap=None,
        pointmap_colors=None,
        mask_colors=None,
        plot_tdfy_kwargs=None,
        title="Tdfy Scene",
    ):
        if mask_colors is None:
            mask_colors = sns.color_palette("husl", len(mask_names))
        if mask_names is None:
            mask_names = [str(i) for i in range(len(pose_targets))]

        cam = SceneVisualizer.camera()
        objects = {}
        for i, mask_name in enumerate(mask_names):
            if mask_name == None:
                continue

            objects[mask_name] = SceneVisualizer.object_pointcloud(
                points_local=pose_targets[i]["xyz_local"].unsqueeze(0),
                quat_l2c=pose_targets[i]["rotation"],
                trans_l2c=pose_targets[i]["translation"],
                scale_l2c=pose_targets[i]["scale"],
                colors=mask_colors[i],
            )

        pointmap_dict = {}
        if pointmap is not None:
            pointmap[pointmap.isnan()] = 0
            pointmap_dict = SceneVisualizer._create_pointmap_structure(
                pointmap=pointmap,
                image=pointmap_colors,
                filter_pointmap_edges=True,
            )

        if plot_tdfy_kwargs is None:
            plot_tdfy_kwargs = copy.deepcopy(NO_BACKGROUND)
        if "height" not in plot_tdfy_kwargs:
            plot_tdfy_kwargs["height"] = 1000
        if "width" not in plot_tdfy_kwargs:
            plot_tdfy_kwargs["width"] = 1000

        fig = plot_tdfy_scene(
            {
                title: {
                    "camera": cam,
                    **objects,
                    **pointmap_dict,
                }
            },
            **plot_tdfy_kwargs,
        )
        return fig

    @staticmethod
    def _create_pointmap_structure(
        pointmap: torch.Tensor,
        image: torch.Tensor,
        show_pointmap_as_mesh: bool = True,
        clip_pointmap_colors_for_vis: bool = True,
        filter_pointmap_edges: bool = True,
    ):
        if pointmap is None:
            return {}

        if show_pointmap_as_mesh:
            if image is None:
                image = torch.zeros_like(pointmap)
            struct = SceneVisualizer.pointmap_to_mesh(
                pointmap=pointmap,
                image=image,
                clip_pointmap_colors_for_vis=clip_pointmap_colors_for_vis,
                filter_edges=filter_pointmap_edges,
            )
            return {"Pointmap mesh": struct}
        else:
            struct = SceneVisualizer.pointmap_to_pointcloud(
                pointmap=pointmap, image=image
            )
            return {"Pointmap pointcloud": struct}

    @staticmethod
    def camera(
        quaternion: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            quaternion: (4,) tensor of quaternion
            translation: (3,) tensor of translation
        """
        if quaternion is None:
            quaternion = torch.tensor([1, 0, 0, 0]).unsqueeze(0)
        if translation is None:
            translation = torch.tensor([0, 0, 0]).unsqueeze(0)
        R = quaternion_to_matrix(quaternion)
        return PerspectiveCameras(R=R, T=translation)

    @staticmethod
    def object_pointcloud(
        points_local: torch.Tensor,
        quat_l2c: torch.Tensor,
        trans_l2c: torch.Tensor,
        scale_l2c: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            points_local: (N, 3) tensor of point coordinates
            colors: (N, 3) tensor of colors
        """
        if colors is None:
            colors = torch.ones_like(points_local) * torch.tensor(
                (1.0, 0.0, 0.0), device=points_local.device
            )
        elif isinstance(colors, tuple):
            colors = torch.ones_like(points_local) * torch.tensor(
                colors, device=points_local.device
            )

        R_l2c = quaternion_to_matrix(quat_l2c)
        l2c_transform = compose_transform(
            scale=scale_l2c, rotation=R_l2c, translation=trans_l2c
        )
        points_world = l2c_transform.transform_points(points_local)
        return Pointclouds(points=points_world, features=colors)

    @staticmethod
    def pointmap_to_pointcloud(pointmap: torch.Tensor, image: torch.Tensor):
        """
        Args:
            pointmap: (H, W, 3) tensor of point coordinates
            image: (H, W, 3) tensor of image
        """
        if image is not None:
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = image.reshape(-1, 3).unsqueeze(0).float()

        return Pointclouds(
            points=pointmap.reshape(-1, 3).unsqueeze(0),
            features=image,
        )

    @staticmethod
    def pointmap_to_mesh(
        pointmap: torch.Tensor,
        image: torch.Tensor,
        clip_pointmap_colors_for_vis: bool = True,
        filter_edges: bool = True,
        clamp_eps: float = 1 / 254,
    ):
        """
        Args:
            pointmap: (H, W, 3) tensor of point coordinates
            image: (H, W, 3) tensor of image
        """
        pointmap = pointmap.cpu().numpy()
        if image is None:
            image = torch.zeros_like(pointmap)
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)

        if clip_pointmap_colors_for_vis:
            # Not sure why, but this is needed to avoid underflow in the visualization
            # We also clip to prevent overflow, just in case and since this is just for visualization
            image = image.clamp(clamp_eps, 1 - clamp_eps)
        image = image.cpu().numpy()
        mesh = mesh_from_pointmap(pointmap, image, filter_edges=filter_edges)
        vertices = torch.from_numpy(mesh.vertices)
        faces = torch.from_numpy(mesh.faces)
        vertex_colors = torch.from_numpy(mesh.vertex_colors)
        return create_textured_mesh(vertices, faces, vertex_colors)
