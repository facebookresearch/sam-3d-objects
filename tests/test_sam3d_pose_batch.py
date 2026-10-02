import os
import unittest

os.environ["LIDRA_SKIP_INIT"] = "1"

import torch
from pytorch3d.transforms import Transform3d

from sam3d_objects.data.dataset.tdfy.pose_target import InstancePose


class PoseBatchBroadcastTest(unittest.TestCase):
    def check_scene_batches(self, b, k, scalar_scale=False):
        channels = 1 if scalar_scale else 3
        scale = torch.arange(1, b * k * channels + 1).float().reshape(b, k, channels) / 5
        rotation = torch.zeros(b, k, 4)
        rotation[..., 0] = 1
        translation = torch.arange(b * k * 3).float().reshape(b, k, 3) / 7
        scene_scale = torch.arange(1, b + 1).float() * 2
        shift = torch.arange(b * 3).float().reshape(b, 3) + 10
        transform = Transform3d().scale(scene_scale).translate(shift)
        result = InstancePose._broadcast_postcompose(scale, rotation, translation, transform)
        expected_scale = scale * scene_scale[:, None, None]
        expected_translation = translation * scene_scale[:, None, None] + shift[:, None]
        torch.testing.assert_close(result[0], expected_scale)
        torch.testing.assert_close(result[1], rotation)
        torch.testing.assert_close(result[2], expected_translation)
        self.assertEqual(result[0].shape, scale.shape)
        self.assertEqual(result[2].shape, translation.shape)

    def test_distinct_scene_transforms_stay_with_each_scene(self):
        self.check_scene_batches(2, 3)
        self.check_scene_batches(3, 2)

    def test_isotropic_object_scale_preserves_shape_and_scene_order(self):
        self.check_scene_batches(2, 4, scalar_scale=True)

    def test_single_scene_and_single_object_are_unchanged(self):
        self.check_scene_batches(1, 5)
        self.check_scene_batches(3, 1)


if __name__ == "__main__":
    unittest.main()
