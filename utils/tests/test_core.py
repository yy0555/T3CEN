"""Synthetic CPU checks; no dataset download or training run is required."""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from modules.hsgroup.hsgconv import GroupBatchNorm2d, GroupConvHS, GroupPool
from modules.hsgroup.transforms import HSVSeparation, TensorReshape
from modules.networks.resnet import BasicBlock, ResNet
from modules.networks.simple_network import HybridZ2CNN, SimpleNetwork, Z2CNN


class CoreSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_lifting_and_group_layers(self):
        image = Image.fromarray(
            np.random.default_rng(123).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        )
        for h, s, l in [(1, 1, 1), (2, 1, 1), (1, 3, 1), (1, 1, 2), (2, 3, 2)]:
            with self.subTest(groups=(h, s, l)):
                group_size = h * s * l
                lifted = TensorReshape()(HSVSeparation(h, s, l)(image))
                self.assertEqual(lifted.shape, (3 * group_size, 8, 8))
                x = torch.randn(2, 3 * group_size, 8, 8, requires_grad=True)

                def shifted(tensor):
                    return (
                        tensor.reshape(2, h, s, l, -1, 8, 8)
                        .roll((1, 1, 1), (1, 2, 3))
                        .reshape_as(tensor)
                    )

                conv = GroupConvHS(
                    3, 4, 3, padding=1,
                    n_groups_hue=h, n_groups_saturation=s, n_groups_luminance=l,
                )
                y = conv(x)
                torch.testing.assert_close(
                    conv(shifted(x)), shifted(y), atol=2e-6, rtol=1e-5
                )
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(torch.isfinite(conv.conv_weight.grad).all())
                norm = GroupBatchNorm2d(3, h, s, l)
                torch.testing.assert_close(
                    norm(shifted(x)), shifted(norm(x)), atol=2e-6, rtol=1e-5
                )
                pool = GroupPool(group_size)
                torch.testing.assert_close(pool(shifted(x)), pool(x))

    def test_resnets_forward_backward_without_writing_files(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for groups in [(1, 1, 1), (2, 2, 2), (2, 3, 1)]:
                    with self.subTest(groups=groups):
                        h, s, l = groups
                        net = ResNet(
                            BasicBlock, [1, 1], num_classes=4, shapes=[4, 8],
                            n_groups_hue=h, n_groups_saturation=s,
                            n_groups_luminance=l,
                        )
                        channels = 3 * h * s * l
                        x = torch.randn(2, channels, 16, 16, requires_grad=True)
                        logits = net(x)
                        self.assertEqual(logits.shape, (2, 4))
                        F.cross_entropy(logits, torch.tensor([1, 2])).backward()
                        self.assertTrue(torch.isfinite(x.grad).all())
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                os.chdir(original_directory)

    def test_small_network_constructor_contract(self):
        for network in [SimpleNetwork, Z2CNN, HybridZ2CNN]:
            for h, s in [(1, 1), (2, 3)]:
                with self.subTest(network=network.__name__, groups=(h, s)):
                    net = network(
                        num_classes=4, n_groups_hue=h, n_groups_saturation=s,
                        n_groups_luminance=1,
                    )
                    with torch.no_grad():
                        result = net(torch.randn(2, 3 * h * s, 28, 28))
                    self.assertEqual(result.shape, (2, 4))
                    with self.assertRaisesRegex(ValueError, "luminance"):
                        network(
                            num_classes=4, n_groups_hue=2, n_groups_saturation=1,
                            n_groups_luminance=2,
                        )


if __name__ == "__main__":
    unittest.main()
