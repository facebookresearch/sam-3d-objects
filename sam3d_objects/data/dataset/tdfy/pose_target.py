# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field, fields as dataclass_fields
from loguru import logger

from sam3d_objects.data.utils import expand_as_right, tree_tensor_map
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform, decompose_transform
from pytorch3d.transforms import Transform3d, quaternion_to_matrix, matrix_to_quaternion


@dataclass
class InstancePose:
    """
    Stores the pose of an object.
    Also, stores some information about the scene that was used to normalize the pose.
    """

    instance_scale_l2c: torch.Tensor
    instance_position_l2c: torch.Tensor
    instance_quaternion_l2c: torch.Tensor
    scene_scale: torch.Tensor
    scene_shift: torch.Tensor

    @classmethod
    def _broadcast_postcompose(
        cls,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        transform_to_postcompose: Transform3d,
    ) -> Transform3d:
        """
        Assumes scale, rotation, translation are of shape:
            B, K, C
            ---
            B: batch size
            K: number of objects
            C: number of channels

        Takes a transform where
            get_matrix() has shape (B, 3, 3)

        Returns pose.compose(transform_to_postcompose)
        """
        scale_c = scale.shape[-1]
        ndim_orig = scale.ndim
        if ndim_orig == 3:
            b, k, _ = scale.shape
        elif ndim_orig == 2:
            b = scale.shape[0]
            k = 1
        elif ndim_orig == 1:
            b = 1
            k = 1
        else:
            raise ValueError(f"Invalid scale shape: {scale.shape}")

        # Create transform of shape (B * K)
        wide = {"scale": scale, "rotation": rotation, "translation": translation}
        shapes_orig = {k: v.shape for k, v in wide.items()}
        long = tree_tensor_map(lambda x: x.reshape(b * k, x.shape[-1]), wide)
        long["rotation"] = quaternion_to_matrix(long["rotation"])
        if scale_c == 1:
            long["scale"] = long["scale"].expand(b * k, 3)

        composed = compose_transform(**long)

        # Apply transform to shape (B * K)
        pc_transform = transform_to_postcompose.get_matrix()
        pc_transform = pc_transform.repeat(k, 1, 1)
        stacked_pc_transform = Transform3d(matrix=pc_transform)
        assert stacked_pc_transform.get_matrix().shape == composed.get_matrix().shape
        postcomposed = composed.compose(stacked_pc_transform)

        # Decompose transform to shape (B, K, C)
        scale, rotation, translation = decompose_transform(postcomposed)
        rotation = matrix_to_quaternion(rotation)
        pc_long = {"scale": scale, "rotation": rotation, "translation": translation}
        pc_wide = tree_tensor_map(lambda x: x.reshape(b, k, x.shape[-1]), pc_long)
        if scale_c == 1:
            pc_wide["scale"] = pc_wide["scale"][..., 0].unsqueeze(-1)
        for k, shape in shapes_orig.items():
            pc_wide[k] = pc_wide[k].reshape(*shape)
        return pc_wide["scale"], pc_wide["rotation"], pc_wide["translation"]


@dataclass
class PoseTarget:
    x_instance_scale: torch.Tensor
    x_instance_rotation: torch.Tensor
    x_instance_translation: torch.Tensor
    x_scene_scale: torch.Tensor
    x_scene_center: torch.Tensor
    x_translation_scale: torch.Tensor
    pose_target_convention: str = field(default="unknown")


@dataclass
class InvariantPoseTarget:
    """
    This is the canonical representation of pose targets, used for computing metrics.
        instance_pose <-> invariant_pose_targets <-> all other pose_target_conventions

    Background:
    ---
    We want to estimate a transformation T: R³ → R³ despite scene scale ambiguity.

    The transformation taking object points to scene points is defined as
        T(x) = s · R(q) · x + t
        where:
            - x is a point in the object coordinate frame,
            - q is a unit quaternion representing rotation,
            - s is the object-to-scene scale, and
            - t is the translation.

    However, there is an inherent scale ambiguity in the scene, denoted as s_scene;
    This ambiguity introduces irreducible error that complicates both evaluation and training.

    To decouple the scene scale from the invariant quantities, we define:
        T(x)  = s_scene · |t_rel| [ s_tilde · R(q) · x + t_unit ]
        where we define
            t_rel = t / s_scene
            s_rel = s / s_scene
            s_tilde = s_rel / |t_rel|
            t_unit = t_rel / |t_rel|

    During training, you would predict (q, s_tilde, t_unit), leaving s_scene separate.


    Hand-wavy error analysis:
    ---
    1. Naive (coupled) estimate:
       T(x) = s_scene [ s_rel · R(q) · x + t_rel ]

       We can define:
           U = ln(s_rel)
           V = ln(|t_rel|)
       so that the error is governed by Var(U + V).

    2. In the decoupled case, we have:
       T(x) = s_scene · |t_rel| [ s_tilde · R(q) · x + t_unit ]
            = s_scene · |t_rel| [ (s_rel / |t_rel|) R(q) · x + t_unit ]
       Then ln(s_tilde) = ln(s_rel) - ln(|t_rel|) = U - V, and the error is
       Var(U - V) = Var(U) + Var(V) - 2Cov(U, V).

    """

    # These are invariant
    q: torch.Tensor
    t_unit: torch.Tensor
    s_scene: torch.Tensor
    t_scene_center: Optional[torch.Tensor] = None
    t_rel_norm: Optional[torch.Tensor] = None
    s_tilde: Optional[torch.Tensor] = None
    s_rel: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Check that fields that are required always have values.
        if self.q is None:
            raise ValueError("Field 'q' (quaternion) must be provided.")
        if self.s_scene is None:
            raise ValueError("Field 's_scene' must be provided.")
        if self.s_rel is None:
            if self.s_tilde is not None:
                self.s_rel = self.s_tilde * self.t_rel_norm
            else:
                raise ValueError("Field 's_rel' or 's_tilde' must be provided.")
        if self.t_unit is None:
            raise ValueError("Field 't_unit' must be provided.")

        if self.t_scene_center is None:
            self.t_scene_center = torch.zeros_like(self.t_unit)

        # There is a simple relationship between s_tilde and t_rel_norm:
        #    s_tilde = s_rel / t_rel_norm
        #
        # If one of these is missing and the other is provided, we can compute the missing field.
        if self.s_tilde is None and self.t_rel_norm is not None:
            self.s_tilde = self.s_rel / self.t_rel_norm
        elif self.t_rel_norm is None and self.s_tilde is not None:
            self.t_rel_norm = self.s_rel / self.s_tilde

        # If both are provided, we check for consistency.
        if self.s_tilde is not None and self.t_rel_norm is not None:
            computed_s_tilde = self.s_rel / self.t_rel_norm
            # If the provided s_tilde deviates from what is computed, update it.
            if not torch.allclose(self.s_tilde, computed_s_tilde, atol=1e-6):
                logger.warning(
                    f"s_tilde and t_rel_norm are provided, but they are not consistent. "
                    f"Updating s_tilde to {computed_s_tilde}."
                )
                self.s_tilde = computed_s_tilde

        self._validate_fields()

    def _validate_fields(self):
        for field in self.__dict__:
            if self.__dict__[field] is None:
                raise ValueError(f"Field '{field}' must be provided.")


    @staticmethod
    def from_instance_pose(instance_pose: InstancePose) -> "InvariantPoseTarget":
        q = instance_pose.instance_quaternion_l2c
        s_obj_to_scene = instance_pose.instance_scale_l2c      # (..., 1) or (..., 3)
        t_obj_to_scene = instance_pose.instance_position_l2c   # (..., 3)
        s_scene = instance_pose.scene_scale                    # (..., 1) or scalar-broadcastable
        t_scene_center = instance_pose.scene_shift             # (..., 3)

        # Normalize to scene scale (per the derivation)
        if not ( s_obj_to_scene.ndim == (s_scene.ndim + 1)):
            raise ValueError(f"s_scene should be ND [...,3] and s_obj_to_scene should be (N+1)D [...,K,3], but got {s_scene.shape=} {s_obj_to_scene.shape=}")
        if not (t_obj_to_scene.ndim == (s_scene.ndim + 1)):
            raise ValueError(f"t_scene_center should be ND [B,3] and t_obj_to_scene should be (N+1)D [B,K,3], but got {t_scene_center.shape=} {t_obj_to_scene.shape=}")
        s_scene_exp = s_scene.unsqueeze(-2)

        s_rel = s_obj_to_scene / s_scene_exp
        t_rel = t_obj_to_scene / s_scene_exp

        # Robust norms
        eps = 1e-8
        t_rel_norm = t_rel.norm(dim=-1, keepdim=True).clamp_min(eps)

        s_tilde = s_rel / t_rel_norm
        t_unit = t_rel / t_rel_norm

        return InvariantPoseTarget(
            q=q,
            s_scene=s_scene,
            t_scene_center=t_scene_center,
            s_rel=s_rel,
            s_tilde=s_tilde,
            t_unit=t_unit,
            t_rel_norm=t_rel_norm,
        )


    @staticmethod
    def to_instance_pose(invariant_targets: "InvariantPoseTarget") -> InstancePose:
        # scale factor per the derivation: s_scene * |t_rel|
        # Normalize to scene scale (per the derivation)
        t_rel_norm_ndim = invariant_targets.t_rel_norm.ndim
        if not (invariant_targets.s_scene.ndim == (t_rel_norm_ndim - 1)) :
            raise ValueError(f"s_scene should be ND [...,3] and t_rel_norm should be (N+1)D [...,K,3], but got {invariant_targets.s_scene.shape=} {invariant_targets.t_rel_norm.shape=}")

        scale = invariant_targets.s_scene.unsqueeze(-2) * invariant_targets.t_rel_norm
        return InstancePose(
            instance_scale_l2c=invariant_targets.s_tilde * scale,
            instance_position_l2c=invariant_targets.t_unit * scale,
            instance_quaternion_l2c=invariant_targets.q,
            scene_scale=invariant_targets.s_scene,
            scene_shift=invariant_targets.t_scene_center,
        )


class PoseTargetConvention:
    """
    Converts pose_targets <-> instance_pose <-> invariant_pose_targets
    """

    pose_target_convention: str

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget) -> PoseTarget:
        raise NotImplementedError("Implement this in a subclass")

    @classmethod
    def to_invariant(cls, instance_pose: InstancePose) -> InvariantPoseTarget:
        raise NotImplementedError("Implement this in a subclass")

    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose) -> PoseTarget:
        invariant_targets = InvariantPoseTarget.from_instance_pose(instance_pose)
        return cls.from_invariant(invariant_targets)

    @classmethod
    def to_instance_pose(cls, pose_target: PoseTarget) -> InstancePose:
        invariant_targets = cls.to_invariant(pose_target)
        return InvariantPoseTarget.to_instance_pose(invariant_targets)


class ScaleShiftInvariant(PoseTargetConvention):
    """

    Midas eq. (6): https://arxiv.org/pdf/1907.01341v3
    But for pointmaps (see MoGe): https://arxiv.org/pdf/2410.19115
    """

    pose_target_convention: str = "ScaleShiftInvariant"
    scale_mean = torch.tensor([1.0232692956924438, 1.0232691764831543, 1.0232692956924438]).to(torch.float32)
    scale_std = torch.tensor([1.3773751258850098, 1.3773752450942993, 1.3773750066757202]).to(torch.float32)
    translation_mean = torch.tensor([0.003191213821992278, 0.017236359417438507, 0.9401122331619263]).to(torch.float32)
    translation_std = torch.tensor([1.341888666152954, 0.7665449380874634, 3.175130605697632]).to(torch.float32)

    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose, normalize: bool = False) -> PoseTarget:
        metric_to_ssi = cls.ssi_to_metric(
            instance_pose.scene_scale, instance_pose.scene_shift
        ).inverse()

        ssi_scale, ssi_rotation, ssi_translation = InstancePose._broadcast_postcompose(
            scale=instance_pose.instance_scale_l2c,
            rotation=instance_pose.instance_quaternion_l2c,
            translation=instance_pose.instance_position_l2c,
            transform_to_postcompose=metric_to_ssi,
        )
        # logger.info(f"{normalize=} {ssi_scale.shape=} {ssi_rotation.shape=} {ssi_translation.shape=}")
        if normalize:
            device = ssi_scale.device
            ssi_scale = (ssi_scale - cls.scale_mean.to(device)) / cls.scale_std.to(device)
            ssi_translation = (ssi_translation - cls.translation_mean.to(device)) / cls.translation_std.to(device)

        return PoseTarget(
            x_instance_scale=ssi_scale,
            x_instance_rotation=ssi_rotation,
            x_instance_translation=ssi_translation,
            x_scene_scale=instance_pose.scene_scale,
            x_scene_center=instance_pose.scene_shift,
            x_translation_scale=torch.ones_like(ssi_scale)[..., 0].unsqueeze(-1),
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_instance_pose(cls, pose_target: PoseTarget, normalize: bool = False) -> InstancePose:
        scene_scale = pose_target.x_scene_scale
        scene_shift = pose_target.x_scene_center
        ssi_to_metric = cls.ssi_to_metric(scene_scale, scene_shift)

        if normalize:
            device = pose_target.x_instance_scale.device
            pose_target.x_instance_scale = pose_target.x_instance_scale * cls.scale_std.to(device) + cls.scale_mean.to(device)
            pose_target.x_instance_translation = pose_target.x_instance_translation * cls.translation_std.to(device) + cls.translation_mean.to(device)

        ins_scale, ins_rotation, ins_translation = InstancePose._broadcast_postcompose(
            scale=pose_target.x_instance_scale,
            rotation=pose_target.x_instance_rotation,
            translation=pose_target.x_instance_translation,
            transform_to_postcompose=ssi_to_metric,
        )

        return InstancePose(
            instance_scale_l2c=ins_scale,
            instance_position_l2c=ins_translation,
            instance_quaternion_l2c=ins_rotation,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget, normalize: bool = False) -> InvariantPoseTarget:
        instance_pose = cls.to_instance_pose(pose_target, normalize=normalize)
        return InvariantPoseTarget.from_instance_pose(instance_pose)

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget, normalize: bool = False) -> PoseTarget:
        instance_pose = InvariantPoseTarget.to_instance_pose(invariant_targets)
        return cls.from_instance_pose(instance_pose, normalize=normalize)

    @classmethod
    def get_scale_and_shift(cls, pointmap):
        shift_z = pointmap[..., -1].nanmedian().unsqueeze(0)
        shift = torch.zeros_like(shift_z.expand(1, 3))
        shift[..., -1] = shift_z

        shifted_pointmap = pointmap - shift
        scale = shifted_pointmap.abs().nanmean().to(shift.device)

        shift = shift.reshape(3)
        scale = scale.expand(3)

        return scale, shift

    @staticmethod
    def ssi_to_metric(scale: torch.Tensor, shift: torch.Tensor):
        if scale.ndim == 1:
            scale = scale.unsqueeze(0)
        if shift.ndim == 1:
            shift = shift.unsqueeze(0)
        return Transform3d().scale(scale).translate(shift).to(shift.device)


class ScaleShiftInvariantWTranslationScale(PoseTargetConvention):
    """

    Midas eq. (6): https://arxiv.org/pdf/1907.01341v3
    But for pointmaps (see MoGe): https://arxiv.org/pdf/2410.19115
    """

    pose_target_convention: str = "ScaleShiftInvariantWTranslationScale"
    scale_mean = torch.tensor([1.0232692956924438, 1.0232691764831543, 1.0232692956924438]).to(torch.float32)
    scale_std = torch.tensor([1.3773751258850098, 1.3773752450942993, 1.3773750066757202]).to(torch.float32)
    translation_mean = torch.tensor([0.003191213821992278, 0.017236359417438507, 0.9401122331619263]).to(torch.float32)
    translation_std = torch.tensor([1.341888666152954, 0.7665449380874634, 3.175130605697632]).to(torch.float32)

    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose, normalize: bool = False) -> PoseTarget:
        metric_to_ssi = cls.ssi_to_metric(
            instance_pose.scene_scale, instance_pose.scene_shift
        ).inverse()

        ssi_scale, ssi_rotation, ssi_translation = InstancePose._broadcast_postcompose(
            scale=instance_pose.instance_scale_l2c,
            rotation=instance_pose.instance_quaternion_l2c,
            translation=instance_pose.instance_position_l2c,
            transform_to_postcompose=metric_to_ssi,
        )

        ssi_translation_scale = ssi_translation.norm(dim=-1, keepdim=True)
        ssi_translation_unit = ssi_translation / ssi_translation_scale.clamp_min(1e-7)

        return PoseTarget(
            x_instance_scale=ssi_scale,
            x_instance_rotation=ssi_rotation,
            x_instance_translation=ssi_translation_unit,
            x_scene_scale=instance_pose.scene_scale,
            x_scene_center=instance_pose.scene_shift,
            x_translation_scale=ssi_translation_scale,
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_instance_pose(cls, pose_target: PoseTarget, normalize: bool = False) -> InstancePose:
        scene_scale = pose_target.x_scene_scale
        scene_shift = pose_target.x_scene_center
        ssi_to_metric = cls.ssi_to_metric(scene_scale, scene_shift)

        ins_translation_unit = pose_target.x_instance_translation / pose_target.x_instance_translation.norm(dim=-1, keepdim=True)
        ins_translation = ins_translation_unit * pose_target.x_translation_scale


        ins_scale, ins_rotation, ins_translation = InstancePose._broadcast_postcompose(
            scale=pose_target.x_instance_scale,
            rotation=pose_target.x_instance_rotation,
            translation=ins_translation,
            transform_to_postcompose=ssi_to_metric,
        )


        return InstancePose(
            instance_scale_l2c=ins_scale,
            instance_position_l2c=ins_translation,
            instance_quaternion_l2c=ins_rotation,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget) -> InvariantPoseTarget:
        instance_pose = cls.to_instance_pose(pose_target)
        return InvariantPoseTarget.from_instance_pose(instance_pose)

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget) -> PoseTarget:
        instance_pose = InvariantPoseTarget.to_instance_pose(invariant_targets)
        return cls.from_instance_pose(instance_pose)

    @classmethod
    def get_scale_and_shift(cls, pointmap):
        shift_z = pointmap[..., -1].nanmedian().unsqueeze(0)
        shift = torch.zeros_like(shift_z.expand(1, 3))
        shift[..., -1] = shift_z

        shifted_pointmap = pointmap - shift
        scale = shifted_pointmap.abs().nanmean().to(shift.device)

        shift = shift.reshape(3)
        scale = scale.expand(3)

        return scale, shift

    @staticmethod
    def ssi_to_metric(scale: torch.Tensor, shift: torch.Tensor):
        if scale.ndim == 1:
            scale = scale.unsqueeze(0)
        if shift.ndim == 1:
            shift = shift.unsqueeze(0)
        return Transform3d().scale(scale).translate(shift).to(shift.device)


class DisparitySpace(PoseTargetConvention):
    pose_target_convention: str = "DisparitySpace"
    
    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose, normalize: bool = False) -> PoseTarget:
 
        # x_instance_scale = orig_scale / scene_scale
        # x_instance_translation = [x/z, y/z, 0]  / scene_scale
        # x_translation_scale = z  / scene_scale
        assert torch.allclose(instance_pose.scene_scale, torch.ones_like(instance_pose.scene_scale))

        if not instance_pose.scene_shift.ndim == instance_pose.instance_position_l2c.ndim - 1:
            raise ValueError(f"scene_shift must be (N+1)D and instance_position_l2c must be (N+1)D, but got {instance_pose.scene_shift.ndim} and {instance_pose.instance_position_l2c.ndim}")
        shift_xy, shift_z_log = instance_pose.scene_shift.unsqueeze(-2).split([2, 1], dim=-1)


        pose_xy, pose_z = instance_pose.instance_position_l2c.split([2, 1], dim=-1)
        # Handle batch dimensions properly
        if shift_xy.ndim < pose_xy.ndim:
            shift_xy = shift_xy.unsqueeze(-2)
        pose_xy_scaled = pose_xy / pose_z - shift_xy

        pose_z_scaled_log = torch.log(pose_z) - shift_z_log
        x_instance_scale_log = torch.log(instance_pose.instance_scale_l2c) - torch.log(pose_z)

        x_instance_translation = torch.cat([pose_xy_scaled, torch.zeros_like(pose_z)], dim=-1)
        x_translation_scale = torch.exp(pose_z_scaled_log)
        x_instance_scale = torch.exp(x_instance_scale_log)



        return PoseTarget(
            x_instance_scale=x_instance_scale,
            x_instance_translation=x_instance_translation,
            x_instance_rotation=instance_pose.instance_quaternion_l2c,
            x_scene_scale=instance_pose.scene_scale,
            x_scene_center=instance_pose.scene_shift,
            x_translation_scale=x_translation_scale,
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_instance_pose(cls, pose_target: PoseTarget, normalize: bool = False) -> InstancePose:
        scene_scale = pose_target.x_scene_scale
        scene_shift = pose_target.x_scene_center

        if not pose_target.x_scene_center.ndim == pose_target.x_instance_translation.ndim - 1:
            raise ValueError(f"x_scene_center must be (N+1)D and x_instance_translation must be (N+1)D, but got {pose_target.x_scene_center.ndim} and {pose_target.x_instance_translation.ndim}")
        shift_xy, shift_z_log = pose_target.x_scene_center.unsqueeze(-2).split([2, 1], dim=-1)
        scene_z_scale = torch.exp(shift_z_log)
 
        z = pose_target.x_translation_scale
        ins_translation = pose_target.x_instance_translation.clone()
        ins_translation[...,2] = 1.0
        ins_translation[...,:2] = ins_translation[...,:2] + shift_xy
        ins_translation = ins_translation * z * scene_z_scale

        ins_scale = pose_target.x_instance_scale * z * scene_z_scale

        return InstancePose(
            instance_scale_l2c=ins_scale * scene_scale,
            instance_position_l2c=ins_translation * scene_scale,
            instance_quaternion_l2c=pose_target.x_instance_rotation,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget, normalize: bool = False) -> InvariantPoseTarget:
        instance_pose = cls.to_instance_pose(pose_target, normalize=normalize)
        return InvariantPoseTarget.from_instance_pose(instance_pose)

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget, normalize: bool = False) -> PoseTarget:
        instance_pose = InvariantPoseTarget.to_instance_pose(invariant_targets)
        return cls.from_instance_pose(instance_pose, normalize=normalize)



class NormalizedSceneScale(PoseTargetConvention):
    """
    x_instance_scale and x_translation_scale are normalized to x_scene_scale
    """

    pose_target_convention: str = "NormalizedSceneScale"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget):
        translation = invariant_targets.t_unit * invariant_targets.t_rel_norm
        return PoseTarget(
            x_instance_scale=invariant_targets.s_rel,
            x_instance_rotation=invariant_targets.q,
            x_instance_translation=translation,
            x_scene_scale=invariant_targets.s_scene,
            x_scene_center=invariant_targets.t_scene_center,
            x_translation_scale=torch.ones_like(invariant_targets.t_rel_norm),
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget):
        t_rel_norm = torch.norm(
            pose_target.x_instance_translation, dim=-1, keepdim=True
        )
        return InvariantPoseTarget(
            s_scene=pose_target.x_scene_scale,
            s_rel=pose_target.x_instance_scale,
            q=pose_target.x_instance_rotation,
            t_unit=pose_target.x_instance_translation / t_rel_norm,
            t_rel_norm=t_rel_norm,
            t_scene_center=pose_target.x_scene_center,
        )


class Naive(PoseTargetConvention):
    pose_target_convention: str = "Naive"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget):
        s_scene = invariant_targets.s_rel * invariant_targets.s_scene
        t_scene = invariant_targets.t_unit * invariant_targets.t_rel_norm
        return PoseTarget(
            x_instance_scale=s_scene,
            x_instance_rotation=invariant_targets.q,
            x_instance_translation=t_scene,
            x_scene_scale=invariant_targets.s_scene,
            x_scene_center=invariant_targets.t_scene_center,
            x_translation_scale=torch.ones_like(invariant_targets.t_rel_norm),
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget):
        s_scene = pose_target.x_scene_scale
        t_rel_norm = torch.norm(
            pose_target.x_instance_translation, dim=-1, keepdim=True
        )
        return InvariantPoseTarget(
            s_scene=s_scene,
            t_scene_center=pose_target.x_scene_center,
            s_rel=pose_target.x_instance_scale / s_scene,
            q=pose_target.x_instance_rotation,
            t_unit=pose_target.x_instance_translation / t_rel_norm,
            t_rel_norm=t_rel_norm,
        )


class NormalizedSceneScaleAndTranslation(PoseTargetConvention):
    """
    x_instance_scale and x_translation_scale are normalized to x_scene_scale
    x_instance_translation is unit
    """

    pose_target_convention: str = "NormalizedSceneScaleAndTranslation"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget):
        return PoseTarget(
            x_instance_scale=invariant_targets.s_rel,
            x_instance_rotation=invariant_targets.q,
            x_instance_translation=invariant_targets.t_unit,
            x_scene_scale=invariant_targets.s_scene,
            x_scene_center=invariant_targets.t_scene_center,
            x_translation_scale=invariant_targets.t_rel_norm,
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget):
        return InvariantPoseTarget(
            s_scene=pose_target.x_scene_scale,
            t_scene_center=pose_target.x_scene_center,
            s_rel=pose_target.x_instance_scale,
            q=pose_target.x_instance_rotation,
            t_unit=pose_target.x_instance_translation,
            t_rel_norm=pose_target.x_translation_scale,
        )


class ApparentSize(PoseTargetConvention):
    pose_target_convention: str = "ApparentSize"

    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget):
        return PoseTarget(
            x_instance_scale=invariant_targets.s_tilde,
            x_instance_rotation=invariant_targets.q,
            x_instance_translation=invariant_targets.t_unit,
            x_scene_scale=invariant_targets.s_scene,
            x_scene_center=invariant_targets.t_scene_center,
            x_translation_scale=invariant_targets.t_rel_norm,
            pose_target_convention=cls.pose_target_convention,
        )

    @classmethod
    def to_invariant(cls, pose_target: PoseTarget):
        return InvariantPoseTarget(
            s_scene=pose_target.x_scene_scale,
            t_scene_center=pose_target.x_scene_center,
            s_tilde=pose_target.x_instance_scale,
            q=pose_target.x_instance_rotation,
            t_unit=pose_target.x_instance_translation,
            t_rel_norm=pose_target.x_translation_scale,
        )


class Identity(PoseTargetConvention):
    """
    Identity convention - no transformation applied.
    Direct passthrough mapping between instance pose and pose target values.
    This preserves all values including scene_scale and scene_shift.
    """
    
    pose_target_convention: str = "Identity"
    
    @classmethod
    def from_instance_pose(cls, instance_pose: InstancePose) -> PoseTarget:
        return PoseTarget(
            x_instance_scale=instance_pose.instance_scale_l2c,
            x_instance_rotation=instance_pose.instance_quaternion_l2c,
            x_instance_translation=instance_pose.instance_position_l2c,
            x_scene_scale=instance_pose.scene_scale,
            x_scene_center=instance_pose.scene_shift,
            x_translation_scale=torch.ones_like(instance_pose.instance_scale_l2c)[..., 0].unsqueeze(-1),
            pose_target_convention=cls.pose_target_convention,
        )
    
    @classmethod
    def to_instance_pose(cls, pose_target: PoseTarget) -> InstancePose:
        return InstancePose(
            instance_scale_l2c=pose_target.x_instance_scale,
            instance_position_l2c=pose_target.x_instance_translation,
            instance_quaternion_l2c=pose_target.x_instance_rotation,
            scene_scale=pose_target.x_scene_scale,
            scene_shift=pose_target.x_scene_center,
        )
    
    @classmethod
    def to_invariant(cls, pose_target: PoseTarget) -> InvariantPoseTarget:
        instance_pose = cls.to_instance_pose(pose_target)
        return InvariantPoseTarget.from_instance_pose(instance_pose)
    
    @classmethod
    def from_invariant(cls, invariant_targets: InvariantPoseTarget) -> PoseTarget:
        instance_pose = InvariantPoseTarget.to_instance_pose(invariant_targets)
        return cls.from_instance_pose(instance_pose)


class PoseTargetConverter:
    @staticmethod
    def pose_target_to_instance_pose(pose_target: PoseTarget, normalize: bool = False) -> InstancePose:
        _convention_class = globals()[pose_target.pose_target_convention]
        if _convention_class == ScaleShiftInvariant:
            return _convention_class.to_instance_pose(pose_target, normalize=normalize)
        else:
            return _convention_class.to_instance_pose(pose_target)

    @staticmethod
    def instance_pose_to_pose_target(
        instance_pose: InstancePose, pose_target_convention: str, normalize: bool = False
    ) -> PoseTarget:
        _convention_class = globals()[pose_target_convention]
        if _convention_class == ScaleShiftInvariant:
            return _convention_class.from_instance_pose(instance_pose, normalize=normalize)
        else:
            return _convention_class.from_instance_pose(instance_pose)

    @staticmethod
    def dicts_instance_pose_to_pose_target(
        pose_target_convention: str,
        **kwargs,
    ):
        instance_pose = InstancePose(**kwargs)
        pose_target = PoseTargetConverter.instance_pose_to_pose_target(
            instance_pose, pose_target_convention
        )
        return asdict(pose_target)

    @staticmethod
    def dicts_pose_target_to_instance_pose(
        **kwargs,
    ):
        pose_target_convention = kwargs.get("pose_target_convention")
        _convention_class = globals()[pose_target_convention]
        assert (
            _convention_class.pose_target_convention == pose_target_convention
        ), f"Normalization name mismatch: {_convention_class.pose_target_convention} != {pose_target_convention}"

        normalize = kwargs.pop("normalize", False)
        pose_target = PoseTarget(**kwargs)
        instance_pose = PoseTargetConverter.pose_target_to_instance_pose(pose_target, normalize)
        return {f.name: getattr(instance_pose, f.name) for f in dataclass_fields(instance_pose)}


class LogScaleShiftNormalizer:
    def __init__(self, shift_log: torch.Tensor = 0.0, scale_log: torch.Tensor = 1.0):
        self.shift_log = shift_log
        self.scale_log = scale_log

    def normalize(self, value: torch.Tensor):
        return torch.log(value) - self.shift_log / self.scale_log

    def denormalize(self, value: torch.Tensor):
        return torch.exp(value * self.scale_log + self.shift_log)