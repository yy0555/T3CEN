"""Synthetic checks for named color-shift evaluation datasets."""

import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets.vision import StandardTransform

from dataloader.evaluation import ShiftedHSVSeparation, make_test_loaders
from modules.hsgroup.transforms import HSVSeparation, TensorReshape


class ImageDataset(Dataset):
    def __init__(self, transform):
        pixels = np.random.default_rng(123).integers(0, 256, (6, 8, 3), dtype=np.uint8)
        self.images = [Image.fromarray(pixels), Image.fromarray(255 - pixels)]
        self.targets = [3, 7]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].copy()
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def pipeline():
    return transforms.Compose([
        transforms.CenterCrop((4, 4)),
        HSVSeparation(1, 1, 1),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
        TensorReshape(),
    ])


class PairedTransformImageDataset(ImageDataset):
    """Exercise the paired transform interface used by OxfordIIITPet."""

    def __init__(self, transform):
        super().__init__(transform)
        self.transforms = StandardTransform(transform, target_transform=lambda value: value + 1)

    def __getitem__(self, index):
        return self.transforms(self.images[index].copy(), self.targets[index])


class EvaluationTests(unittest.TestCase):
    def test_variants_shift_only_the_named_channel_and_keep_pipeline_order(self):
        dataset = ImageDataset(pipeline())
        before, label = dataset[0]
        loaders = make_test_loaders(
            dataset, ["luminance", "original", "saturation"], 2, 0
        )
        self.assertEqual([loader.test_shift for loader in loaders],
                         ["luminance", "original", "saturation"])
        clean_hsv = np.asarray(
            transforms.CenterCrop((4, 4))(dataset.images[0]).convert("HSV")
        )
        for loader in loaders:
            with self.subTest(shift=loader.test_shift):
                expected = clean_hsv.astype(np.int16)
                if loader.test_shift != "original":
                    channel = 1 if loader.test_shift == "saturation" else 2
                    expected[:, :, channel] = np.maximum(expected[:, :, channel] - 127, 0)
                expected = torch.tensor(expected, dtype=torch.float32).permute(2, 0, 1)
                expected = (expected / 255 - 0.5) / 0.5
                images, labels = next(iter(loader))
                torch.testing.assert_close(images[0], expected, rtol=0, atol=0)
                self.assertEqual(images.shape, (2, 3, 4, 4))
                self.assertEqual(labels.tolist(), dataset.targets)
                self.assertIs(loader.dataset.images, dataset.images)
                self.assertIs(loader.dataset.targets, dataset.targets)
                self.assertIsNot(loader.dataset.transform, dataset.transform)
        torch.testing.assert_close(dataset[0][0], before, rtol=0, atol=0)
        self.assertEqual(dataset[0][1], label)
        self.assertIsInstance(dataset.transform.transforms[1], HSVSeparation)
        shifted = loaders[0].dataset.transform.transforms[1]
        self.assertIsInstance(shifted, ShiftedHSVSeparation)
        shifted.lifting.amt = 50
        self.assertEqual(dataset.transform.transforms[1].amt, 0)
        self.assertEqual(loaders[1].dataset.transform.transforms[1].amt, 0)

    def test_nested_subsets_preserve_indices_labels_and_data_storage(self):
        dataset = ImageDataset(pipeline())
        subset = Subset(Subset(dataset, [1, 0]), [0])
        loaders = make_test_loaders(subset, ["original", "saturation"], 1, 0)
        for loader in loaders:
            self.assertIsNot(loader.dataset, subset)
            self.assertIsNot(loader.dataset.dataset, subset.dataset)
            self.assertIs(loader.dataset.indices, subset.indices)
            self.assertIs(loader.dataset.dataset.indices, subset.dataset.indices)
            self.assertIs(loader.dataset.dataset.dataset.images, dataset.images)
            self.assertEqual(next(iter(loader))[1].tolist(), [7])
        torch.testing.assert_close(next(iter(loaders[0]))[0][0], subset[0][0])

    def test_paired_vision_dataset_transform_receives_the_shift(self):
        dataset = PairedTransformImageDataset(pipeline())
        before = dataset[0]
        original, saturation = make_test_loaders(
            dataset, ["original", "saturation"], 2, 0
        )
        clean_images, clean_labels = next(iter(original))
        shifted_images, shifted_labels = next(iter(saturation))
        self.assertEqual(clean_labels.tolist(), [4, 8])
        torch.testing.assert_close(clean_labels, shifted_labels)
        torch.testing.assert_close(clean_images[:, [0, 2]], shifted_images[:, [0, 2]])
        self.assertFalse(torch.equal(clean_images[:, 1], shifted_images[:, 1]))
        self.assertIsNot(saturation.dataset.transforms, dataset.transforms)
        self.assertIs(dataset.transforms.transform, dataset.transform)
        torch.testing.assert_close(dataset[0][0], before[0], rtol=0, atol=0)

    def test_shift_validation_and_missing_lifting(self):
        dataset = ImageDataset(pipeline())
        for shifts in [["saturaton"], ["hue"], [], "original"]:
            with self.subTest(shifts=shifts), self.assertRaises(ValueError):
                make_test_loaders(dataset, shifts, 1, 0)
        dataset.transform = transforms.ToTensor()
        with self.assertRaisesRegex(ValueError, "exactly one HSVSeparation"):
            make_test_loaders(dataset, ["saturation"], 1, 0)


if __name__ == "__main__":
    unittest.main()
