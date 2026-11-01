"""Build clean and color-shifted evaluation loaders without copying image data."""

from copy import copy, deepcopy

from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.vision import StandardTransform
from torchvision.transforms import Compose

from modules.hsgroup.transforms import HSVSeparation


TEST_SHIFTS = ("original", "saturation", "luminance")


class ShiftedHSVSeparation:
    """Subtract half the 8-bit channel range immediately before color lifting.

    The integer shift is 255 // 2, matching the existing saturation-shift
    convention. Keep the shifted image in HSV mode to avoid an intervening RGB conversion.
    """

    def __init__(self, lifting, shift):
        if shift not in ("saturation", "luminance"):
            raise ValueError(f"Unsupported color shift: {shift!r}")
        self.lifting = lifting
        self.shift = shift

    def __call__(self, image):
        channels = list(image.convert("HSV").split())
        channel_index = 1 if self.shift == "saturation" else 2
        channels[channel_index] = channels[channel_index].point(
            lambda value: max(0, value - 255 // 2)
        )
        return self.lifting(Image.merge("HSV", tuple(channels)))


def _replace_lifting(transform, shift):
    """Replace the lifting in an already-copied pipeline, preserving its order."""
    if isinstance(transform, HSVSeparation):
        return (
            transform if shift == "original" else ShiftedHSVSeparation(transform, shift),
            1,
        )
    if isinstance(transform, Compose):
        replacements = [_replace_lifting(item, shift) for item in transform.transforms]
        transform.transforms = [item for item, _ in replacements]
        return transform, sum(count for _, count in replacements)
    return transform, 0


def _evaluation_dataset(dataset, shift):
    result = copy(dataset)
    if isinstance(dataset, Subset):
        result.dataset = _evaluation_dataset(dataset.dataset, shift)
    else:
        # Only transforms are deep-copied: image storage, samples, targets and
        # Subset indices remain shared with the original dataset.
        transform, count = _replace_lifting(
            deepcopy(getattr(dataset, "transform", None)), shift
        )
        if count != 1:
            raise ValueError(
                "Evaluation requires exactly one HSVSeparation in dataset.transform; "
                f"found {count}."
            )
        result.transform = transform
        # OxfordIIITPet calls the paired VisionDataset wrapper instead of
        # self.transform directly. Copy it so the original stays unmodified.
        if isinstance(getattr(dataset, "transforms", None), StandardTransform):
            result.transforms = copy(dataset.transforms)
            result.transforms.transform = transform
    return result


def make_test_loaders(dataset, test_shifts, batch_size, num_workers):
    """Return evaluation loaders in the requested shift order.

    Each loader exposes its name as ``test_shift``. Dataset wrappers and
    transform pipelines are independent; the underlying image data is shared.
    Validation can continue using the original dataset unchanged.
    """
    if isinstance(test_shifts, str):
        raise ValueError("test_shifts must be a sequence of shift names, not a string.")
    shifts = list(test_shifts)
    if not shifts:
        raise ValueError("test_shifts must contain at least one shift name.")
    for shift in shifts:
        if shift not in TEST_SHIFTS:
            raise ValueError(f"Unknown test shift {shift!r}; choose from {TEST_SHIFTS}.")

    loaders = []
    for shift in shifts:
        loader = DataLoader(
            _evaluation_dataset(dataset, shift),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
        loader.test_shift = shift
        loaders.append(loader)
    return loaders
