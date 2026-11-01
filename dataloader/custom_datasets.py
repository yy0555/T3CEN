from torch.utils.data import Dataset
import os

import h5py
from PIL import Image
import torch
import torchvision
import numpy as np

class HSVMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, hue_low, hue_high, sat_low=1.0, sat_high=1.0, value_low=1.0, value_high=1.0, train=True, transform=None, download=False):
        """
        Args:
        root (string): Root directory of dataset where `MNIST/processed/training.pt`
            and  `MNIST/processed/test.pt` exist.
            hue_low (float): Lower bound for hue value (between 0 and 1 where 0 is red)
            hue_high (float): Upper bound for hue value (between 0 and 1)
            sat_low (float): Lower bound for saturation value (between 0 and 1)
            sat_high (float): Upper bound for saturation value (between 0 and 1)
            value_low (float): Lower bound for value (brightness) value (between 0 and 1)
            value_high (float): Upper bound for value (brightness) value (between 0 and 1)"""
        super(HSVMNIST, self).__init__(root, train=train, transform=None, download=download)
        self.custom_transform = transform

        self.hue_low = hue_low
        self.hue_high = hue_high

        self.sat_low = sat_low
        self.sat_high = sat_high

        self.value_low = value_low
        self.value_high = value_high

        dataset_len = len(self)

        # create random array of colors based on color codes
        self.random_hue = (np.random.uniform(hue_low, hue_high, (dataset_len)))                  # (N,)
        self.random_sat = (np.random.uniform(sat_low, sat_high, (dataset_len)))                  # (N,)
        self.random_value = (np.random.uniform(value_low, value_high, (dataset_len)))             # (N,)

    def change_color(self, image : Image, index):
        # get RGB values from hue and saturation
        hue = self.random_hue[index] % 1.0
        sat = self.random_sat[index]
        value = self.random_value[index]

        image = image.convert("HSV")
        image = np.array(image)
        # mask = image[:, :, 2] > 0
        image[:, :, 0] = hue * 255
        image[:, :, 1] = sat * 255
        image[:, :, 2] = image[:, :, 2] * value
        image = Image.fromarray(image, mode="HSV").convert("RGB")
        return image
    
    def __getitem__(self, index):
        inputs, labels = super(HSVMNIST, self).__getitem__(index)
        inputs = self.change_color(inputs, index)
        if self.custom_transform:
            inputs = self.custom_transform(inputs)
        return inputs, labels


class SmallNorbDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        self.length = len(data)
    
    def __getitem__(self, index):
        x = self.data[index]

        
        if self.transform:
            x = self.transform(x)
        
        y = self.targets[index]
        
        return x, y

    def __len__(self):
        return self.length


# dataset classes
class HDF5Dataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)  # Number of samples in the HDF5 file

    def __getitem__(self, idx):
        sample_data = self.imgs[idx]  # Access the dataset at the specified index
        data = Image.fromarray(sample_data)
        sample_label = self.labels[idx]  # Access the dataset at the specified index
        label = torch.tensor(sample_label, dtype=torch.long)  # Assuming 'label' is a dataset in your HDF5 file

        if self.transform:
            data = self.transform(data)
        l = label[4]
        return data, l


class Shapes3DDataset(Dataset):
    """Read selected HDF5 rows lazily with a separate handle in each worker."""

    def __init__(self, path, indices, labels, transform=None):
        self.path = str(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels)
        self.transform = transform
        self._handle = None
        self._pid = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self._handle is None or self._pid != os.getpid():
            if self._handle is not None:
                self._handle.close()
            self._handle = h5py.File(self.path, "r")
            self._pid = os.getpid()
        image = Image.fromarray(self._handle["images"][self.indices[index]])
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(int(self.labels[index, 4]), dtype=torch.long)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
