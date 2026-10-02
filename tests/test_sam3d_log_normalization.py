import os
import unittest

os.environ["LIDRA_SKIP_INIT"] = "1"

import torch

from sam3d_objects.data.dataset.tdfy.pose_target import LogScaleShiftNormalizer


class LogScaleShiftNormalizerTest(unittest.TestCase):
    def test_non_unit_scale_matches_log_affine_definition(self):
        values = torch.tensor([.25, 1., 4.], dtype=torch.float64)
        normalizer = LogScaleShiftNormalizer(shift_log=1.5, scale_log=2.5)
        expected = (torch.log(values) - 1.5) / 2.5
        torch.testing.assert_close(normalizer.normalize(values), expected)
        torch.testing.assert_close(normalizer.denormalize(normalizer.normalize(values)), values)

    def test_per_channel_parameters_broadcast_and_roundtrip(self):
        values = torch.tensor([[.5, 2., 8.], [1., 4., 16.]], dtype=torch.float64)
        shift = torch.tensor([-.5, 0., 1.], dtype=torch.float64)
        scale = torch.tensor([.25, 2., 3.], dtype=torch.float64)
        normalizer = LogScaleShiftNormalizer(shift, scale)
        torch.testing.assert_close(normalizer.normalize(values), (torch.log(values) - shift) / scale)
        torch.testing.assert_close(normalizer.denormalize(normalizer.normalize(values)), values)

    def test_inverse_chain_gradient_is_identity(self):
        values = torch.tensor([.1, 1., 10.], dtype=torch.float64, requires_grad=True)
        normalizer = LogScaleShiftNormalizer(shift_log=.7, scale_log=1.8)
        recovered = normalizer.denormalize(normalizer.normalize(values))
        recovered.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))


if __name__ == "__main__":
    unittest.main()
