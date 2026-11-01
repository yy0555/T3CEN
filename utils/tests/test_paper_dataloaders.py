"""No-download checks for the data protocols described by the paper."""

import io
import itertools
import pickle
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
import torch
import torchvision

from dataloader import dataloaders as loaders


class PaperShapesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "3dshapes.h5"
        self.labels = np.array([
            [floor, wall, obj, 1, shape, orientation]
            for floor, wall, obj, shape, orientation in itertools.product(range(2), repeat=5)
        ], dtype=np.float32)
        self.images = np.stack([
            np.full((8, 8, 3), index, dtype=np.uint8) for index in range(len(self.labels))
        ])
        self.write(self.source, self.images)
        self.standard = patch.object(loaders, "DATA_PATH_SHAPES3D_SAT_STANDARD", self.source)
        self.shifted = patch.object(loaders, "DATA_PATH_SHAPES3D_SATS", self.root / "validation")
        self.standard.start()
        self.shifted.start()
        self.addCleanup(self.standard.stop)
        self.addCleanup(self.shifted.stop)

    def write(self, path, images, labels=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as dataset:
            dataset["images"] = images
            dataset["labels"] = self.labels if labels is None else labels

    def get(self, protocol):
        return loaders.get_shapes3d(
            n_groups_hue=1, n_groups_saturation=1, n_groups_luminance=1,
            protocol=protocol, train_fraction=0.5, split_seed=1999,
            batch_size=2, num_workers=0,
        )

    def test_hue_protocol_uses_warm_cold_and_mixed_conditions(self):
        result = self.get("hue")
        train = result.train.dataset
        original, cold, mixed = [loader.dataset for loader in result.test]
        self.assertTrue(np.all(train.labels[:, :3] == [0, 0, 0]))
        self.assertTrue(np.all(original.labels[:, :3] == [0, 0, 0]))
        self.assertTrue(np.all(cold.labels[:, :3] == [1, 1, 1]))
        self.assertTrue(np.all(mixed.labels[:, :3] == [0, 0, 1]))
        self.assertFalse(set(train.indices) & set(original.indices))
        self.assertIs(result.val, result.test[0])
        for other in (cold, mixed):
            np.testing.assert_array_equal(original.labels[:, 3:], other.labels[:, 3:])
        np.testing.assert_array_equal(train.indices, self.get("hue").train.dataset.indices)
        inputs, targets = next(iter(result.train))
        self.assertEqual(tuple(inputs.shape), (2, 3, 8, 8))
        self.assertEqual(tuple(targets.shape), (2,))

    def test_shifted_protocols_keep_heldout_rows_and_sorted_file_order(self):
        for protocol in ("saturation", "hsl"):
            with self.subTest(protocol=protocol):
                directory = self.root / "validation" / protocol
                self.write(directory / "z_second.h5", self.images + 80)
                self.write(directory / "a_first.h5", self.images + 40)
                result = self.get(protocol)
                self.assertEqual(len(result.train.dataset), len(self.labels) // 2)
                self.assertEqual([Path(item.dataset.path).name for item in result.test],
                                 ["3dshapes.h5", "a_first.h5", "z_second.h5"])
                heldout = result.val.dataset.indices
                self.assertFalse(set(result.train.dataset.indices) & set(heldout))
                for test_loader in result.test:
                    np.testing.assert_array_equal(test_loader.dataset.indices, heldout)
                sample = result.test[1].dataset
                self.assertIsNone(sample._handle)
                sample[0]
                restored = pickle.loads(pickle.dumps(sample))
                self.assertIsNone(restored._handle)
                self.assertTrue(torch.equal(sample[0][0], restored[0][0]))

    def test_shifted_files_require_source_alignment(self):
        directory = self.root / "validation" / "saturation"
        self.write(directory / "misaligned.h5", self.images, self.labels[::-1])
        with self.assertRaisesRegex(ValueError, "source row count, labels, and order"):
            self.get("saturation")

    def test_missing_shifted_files_have_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "validation/hsl/\\*.h5"):
            self.get("hsl")


class PaperMNISTTests(unittest.TestCase):
    @staticmethod
    def fake_mnist_init(dataset, root, train=True, transform=None, download=False):
        size = 20 if train else 10
        dataset.data = torch.arange(size * 28 * 28).remainder(256).to(torch.uint8).reshape(size, 28, 28)
        dataset.targets = torch.arange(size).remainder(10)
        dataset.transform = transform
        dataset.target_transform = None

    def test_hue_ranges_validation_split_and_z2cnn_input_shape(self):
        with patch.object(torchvision.datasets.MNIST, "__init__", self.fake_mnist_init):
            np.random.seed(1999)
            result = loaders.get_hue_mnist(n_groups_hue=4, batch_size=2)
            trainset = result.train.dataset
            valset = result.val.dataset
            testset = result.test.dataset
            self.assertEqual((len(trainset), len(valset), len(testset)), (18, 2, 10))
            self.assertFalse(set(trainset.indices) & set(valset.indices))
            self.assertTrue(np.all((trainset.dataset.random_hue >= 0) & (trainset.dataset.random_hue < 1 / 3)))
            self.assertTrue(np.all((testset.random_hue >= 1 / 3) & (testset.random_hue < 1)))
            inputs, targets = next(iter(result.train))
            self.assertEqual(tuple(inputs.shape), (2, 12, 28, 28))
            self.assertTrue(torch.isfinite(inputs).all())
            self.assertEqual(tuple(targets.shape), (2,))
            repeated = loaders.get_hue_mnist(n_groups_hue=4, batch_size=2)
            self.assertEqual(trainset.indices, repeated.train.dataset.indices)
            self.assertIs(loaders.parse_dataloader("hue_mnist"), loaders.get_hue_mnist)


class PaperSplitTests(unittest.TestCase):
    def test_camelyon_uses_distinct_validation_and_test_subsets(self):
        class WILDSDataset:
            def get_subset(self, split, transform=None):
                return split

        # Stub the optional package as well: importing WILDS can start an
        # unrelated package-version network check in its dependencies.
        wilds = SimpleNamespace(get_dataset=lambda *args, **kwargs: WILDSDataset())
        wilds_loaders = SimpleNamespace(
            get_train_loader=lambda name, dataset, **kwargs: ("train_loader", dataset),
            get_eval_loader=lambda name, dataset, **kwargs: ("eval_loader", dataset),
        )
        with patch.dict(sys.modules, {"wilds": wilds, "wilds.common.data_loaders": wilds_loaders}):
            result = loaders.get_camelyon17()
        self.assertEqual(result.train, ("train_loader", "train"))
        self.assertEqual(result.val, ("eval_loader", "val"))
        self.assertEqual(result.test, ("eval_loader", "test"))

    def test_smallnorb_test_order_is_mid_low_high(self):
        import pyarrow as pa

        stream = io.BytesIO()
        Image.new("L", (8, 8), 128).save(stream, format="PNG")
        table = pa.Table.from_pylist([
            {"image_lt": {"bytes": stream.getvalue()}, "category": lighting % 5, "lighting": lighting}
            for lighting in range(6)
        ])
        with patch("pyarrow.parquet.read_table", return_value=table):
            result = loaders.get_smallnorb(batch_size=2, num_workers=0)
        self.assertEqual(result.train.dataset.targets.tolist(), [2, 3])
        self.assertEqual([item.dataset.targets.tolist() for item in result.test], [[2, 3], [0, 1], [4, 0]])
        self.assertIs(result.val, result.test[0])


class PaperRealDatasetTests(unittest.TestCase):
    def test_validation_stays_clean_when_only_luminance_test_is_requested(self):
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, root, transform=None, **kwargs):
                self.transform = transform

            def __len__(self):
                return 12

            def __getitem__(self, index):
                image = Image.new("HSV", (32, 32), (80, 200, 240)).convert("RGB")
                return self.transform(image), index % 2

        for name, factory in [
            ("CIFAR10", loaders.get_cifar), ("CIFAR100", loaders.get_cifar100),
            ("Caltech101", loaders.get_caltech101), ("OxfordIIITPet", loaders.get_pets),
            ("StanfordCars", loaders.get_cars), ("STL10", loaders.get_STL10),
        ]:
            with self.subTest(dataset=name), patch.object(torchvision.datasets, name, SyntheticDataset):
                result = factory(n_groups_hue=1, n_groups_saturation=1, n_groups_luminance=1,
                                 batch_size=2, num_workers=0, test_shifts=("luminance",))
                self.assertEqual(len(result.test), 1)
                self.assertEqual(result.test[0].test_shift, "luminance")
                clean, _ = next(iter(result.val))
                shifted, _ = next(iter(result.test[0]))
                self.assertEqual(clean.shape, shifted.shape)
                self.assertFalse(torch.equal(clean, shifted))
                again, _ = next(iter(result.val))
                self.assertTrue(torch.equal(clean, again))


if __name__ == "__main__":
    unittest.main()
