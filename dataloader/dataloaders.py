import torch
import math
import torchvision
import torchvision.transforms as transforms

from modules.hsgroup.transforms import TensorReshape, HSVSeparation
from dataloader.custom_datasets import HSVMNIST, Shapes3DDataset, SmallNorbDataset
from dataloader.evaluation import make_test_loaders

from collections import namedtuple
import numpy as np
import h5py
import io
from PIL import Image
from utils.constants import DATA_DIR
dataloaders = namedtuple("dataloaders", ["train", "val", "test"])

DATA_PATH_SHAPES3D_SAT_STANDARD = DATA_DIR / "shapes3d" / "3dshapes.h5"
DATA_PATH_SHAPES3D_SATS = DATA_DIR / "shapes3d" / "validation"
DATA_PATH_SMALLNORB = DATA_DIR / "smallnorb"
DATA_NAME_SMALLNORB_TEST = "smallnorb_test.parquet"
DATA_NAME_SMALLNORB_TRAIN = "smallnorb_train.parquet"

def get_shapes3d(
    n_groups_hue=3, n_groups_saturation=3, n_groups_luminance=1, batch_size=128,
    protocol="hue", train_fraction=0.8, split_seed=1999, num_workers=24,
):
    """Paper Shapes protocols with a reproducible, explicitly chosen train split.

    Shifted files must contain all source rows, in the same order, with the
    original labels. Saturation and HSL distributions are supplied by those
    files: this loader does not invent the paper's unspecified shift sampling.
    """
    if protocol not in {"hue", "saturation", "hsl"}:
        raise ValueError("Shapes protocol must be 'hue', 'saturation', or 'hsl'.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one.")
    source = DATA_PATH_SHAPES3D_SAT_STANDARD
    with h5py.File(source, "r") as dataset:
        labels = dataset["labels"][()]
        if labels.ndim != 2 or labels.shape[1] != 6 or len(dataset["images"]) != len(labels):
            raise ValueError(f"{source} must contain matching images[N,...] and labels[N,6].")

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        HSVSeparation(n_groups_hue, n_groups_saturation, n_groups_luminance),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
        TensorReshape(),
    ])
    transform_test = transforms.Compose([
        HSVSeparation(n_groups_hue, n_groups_saturation, n_groups_luminance),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
        TensorReshape(),
    ])

    # Match each warm/cold subset's factor ordering without assuming image size
    # or the full 480,000-row Cartesian-product storage layout.
    if protocol == "hue":
        low = []
        high = []
        for axis in range(3):
            levels = np.unique(labels[:, axis])
            if len(levels) < 2 or len(levels) % 2:
                raise ValueError("Hue protocol requires an even number of levels on each hue axis.")
            midpoint = len(levels) // 2
            low.append(np.isin(labels[:, axis], levels[:midpoint]))
            high.append(np.isin(labels[:, axis], levels[midpoint:]))
        masks = [low[0] & low[1] & low[2], high[0] & high[1] & high[2], low[0] & low[1] & high[2]]
        subsets = []
        for mask in masks:
            indices = np.flatnonzero(mask)
            factor_order = np.lexsort(tuple(labels[indices, axis] for axis in reversed(range(6))))
            subsets.append(indices[factor_order])
        if not len(subsets[0]) or any(len(indices) != len(subsets[0]) for indices in subsets):
            raise ValueError("Hue protocol requires equally sized warm, cold, and mixed subsets.")
        source_indices = subsets[0]
    else:
        source_indices = np.arange(len(labels))

    permutation = np.random.default_rng(split_seed).permutation(len(source_indices))
    split = int(len(source_indices) * train_fraction)
    if split == 0 or split == len(source_indices):
        raise ValueError("Shapes split must leave at least one training and held-out image.")
    train_positions = np.sort(permutation[:split])
    heldout_positions = np.sort(permutation[split:])
    train_indices = source_indices[train_positions]
    heldout_indices = source_indices[heldout_positions]

    def loader(path, indices, transform, shuffle=False):
        dataset = Shapes3DDataset(path, indices, labels[indices], transform)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )

    trainloader = loader(source, train_indices, transform_train, shuffle=True)
    clean_loader = loader(source, heldout_indices, transform_test)
    tests = [clean_loader]
    if protocol == "hue":
        tests.extend(loader(source, subset[heldout_positions], transform_test) for subset in subsets[1:])
    else:
        directory = DATA_PATH_SHAPES3D_SATS / protocol
        paths = sorted(path for path in directory.glob("*.h5") if path.is_file())
        if not paths:
            raise FileNotFoundError(
                f"Place pre-generated {protocol} evaluation files in {directory}/*.h5. "
                "Each file must contain images and the unchanged source labels in source row order."
            )
        for path in paths:
            with h5py.File(path, "r") as shifted:
                if len(shifted["images"]) != len(labels) or not np.array_equal(shifted["labels"][()], labels):
                    raise ValueError(f"{path} must preserve the source row count, labels, and order.")
            tests.append(loader(path, heldout_indices, transform_test))
    return dataloaders(train=trainloader, val=clean_loader, test=tests)


def get_cifar(n_groups_hue = 3, n_groups_saturation=3, n_groups_luminance=1, batch_size=128, test_shifts=("original", "saturation", "luminance"), num_workers=1):
    transform_train = transforms.Compose([

        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape(),
    ])

    transform_test = transforms.Compose([

        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape(),
    ])

    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloaders(
        train=trainloader, val=testloader,
        test=make_test_loaders(testset, test_shifts, batch_size, num_workers),
    )

def get_cifar100(n_groups_hue = 1, n_groups_saturation =1,n_groups_luminance = 1, batch_size=128, test_shifts=("original", "saturation", "luminance"), num_workers=24):
    transform_train = transforms.Compose([

        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape(),
    ])

    transform_test = transforms.Compose([

        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape(),
    ])


    trainset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloaders(
        train=trainloader, val=testloader,
        test=make_test_loaders(testset, test_shifts, batch_size, num_workers),
    )

def get_smallnorb(n_groups_hue = 1,n_groups_saturation=1, n_groups_luminance = 1, batch_size=128, frac_space=1.0, num_workers=24):
    import pyarrow.parquet as pq

    train_split = True
    data_train = []
    labels_train = []
    table = pq.read_table(DATA_PATH_SMALLNORB / DATA_NAME_SMALLNORB_TRAIN)
    for i in range(len(table)):
        image = Image.open(io.BytesIO(table['image_lt'][i][0].as_py()))
        label = table['category'][i].as_py()
        lighting = table['lighting'][i].as_py()
        if not train_split or (lighting > 1 and lighting < 4):
            data_train.append(image)
            labels_train.append(label)

    data_test_lowlight = []
    labels_test_lowlight = []
    data_test_mid = []
    labels_test_mid = []
    data_test_highlight = []
    labels_test_highlight = []

    table = pq.read_table(DATA_PATH_SMALLNORB / DATA_NAME_SMALLNORB_TEST)
    for i in range(len(table)):
        image = Image.open(io.BytesIO(table['image_lt'][i][0].as_py()))
        label = table['category'][i].as_py()
        lighting = table['lighting'][i].as_py()

        if lighting > 3:
            data_test_highlight.append(image)
            labels_test_highlight.append(label)
        elif lighting < 2:
            data_test_lowlight.append(image)
            labels_test_lowlight.append(label)
        else:
            data_test_mid.append(image)
            labels_test_mid.append(label)


    transform_train = transforms.Compose(
        [
            HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )
    transform_test = transforms.Compose(
        [
            HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )

    dataset_train = SmallNorbDataset(data_train, labels_train, transform=transform_train)
    dataset_test_lowlight = SmallNorbDataset(data_test_lowlight, labels_test_lowlight, transform=transform_test)
    dataset_test_highlight = SmallNorbDataset(data_test_highlight, labels_test_highlight, transform=transform_test)
    dataset_test_mid = SmallNorbDataset(data_test_mid, labels_test_mid, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader_lowlight = torch.utils.data.DataLoader(dataset_test_lowlight, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader_highlight = torch.utils.data.DataLoader(dataset_test_highlight, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader_mid = torch.utils.data.DataLoader(dataset_test_mid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloaders(train=trainloader, val=testloader_mid, test=[testloader_mid, testloader_lowlight, testloader_highlight])


def get_caltech101(n_groups_hue=1, n_groups_saturation=1, n_groups_luminance=1, batch_size=128, test_shifts=("original", "saturation", "luminance"), num_workers=24):
    # 1. 定义 Train 和 Test 的 Transform
    transform_train = transforms.Compose([

        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation, n_groups_luminance=n_groups_luminance),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape()
    ])
    # Use the same deterministic crop for clean and shifted evaluation.
    transform_test = transforms.Compose([

        transforms.Resize(256),
        transforms.CenterCrop(224),
        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation, n_groups_luminance=n_groups_luminance),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape()
    ])

    # 2. 实例化两个数据集对象（一个用于取训练集，一个用于取测试集）
    full_set_train = torchvision.datasets.Caltech101(root=DATA_DIR, download=False, transform=transform_train)
    full_set_test = torchvision.datasets.Caltech101(root=DATA_DIR, download=False, transform=transform_test)

    # 3. 使用相同的随机种子进行划分，确保 trainset 和 testset 的样本不重叠且覆盖全体
    train_size = math.floor(0.67 * len(full_set_train))
    test_size = len(full_set_train) - train_size

    # 必须手动指定 generator 以保证两次 random_split 结果完全一致
    trainset, _ = torch.utils.data.random_split(full_set_train, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    _, testset = torch.utils.data.random_split(full_set_test, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # 4. 构建 DataLoader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloaders(
        train=trainloader, val=testloader,
        test=make_test_loaders(testset, test_shifts, batch_size, num_workers),
    )


def get_pets(n_groups_hue = 1, n_groups_saturation = 1,n_groups_luminance = 1, batch_size=128, test_shifts=("original", "saturation", "luminance"), num_workers=24):
    transform_train = transforms.Compose([

        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        TensorReshape()
    ])
    transform_test = transforms.Compose([

        transforms.Resize(256),
        transforms.CenterCrop(224),
        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        TensorReshape()
    ])

    trainset = torchvision.datasets.OxfordIIITPet(root=DATA_DIR, split="trainval", transform=transform_train, download=True)
    test_set = torchvision.datasets.OxfordIIITPet(root=DATA_DIR, split="test", transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloaders(
        train=trainloader, val=testloader,
        test=make_test_loaders(test_set, test_shifts, batch_size, num_workers),
    )

def get_cars(n_groups_hue=1, n_groups_saturation=1, n_groups_luminance=1, batch_size=128, test_shifts=("original", "saturation", "luminance"), num_workers=24):
    transform_train = transforms.Compose([

        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape()
    ])
    transform_test = transforms.Compose([

        transforms.Resize(256),
        transforms.CenterCrop(224),
        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape()
    ])

    trainset = torchvision.datasets.StanfordCars(root=DATA_DIR, split="train", transform=transform_train, download=False)
    testset = torchvision.datasets.StanfordCars(root=DATA_DIR, split="test", transform=transform_test, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloaders(
        train=trainloader, val=testloader,
        test=make_test_loaders(testset, test_shifts, batch_size, num_workers),
    )

def get_STL10(n_groups_hue = 1,n_groups_saturation=1, n_groups_luminance = 1, batch_size=128, test_shifts=("original", "saturation", "luminance"), num_workers=24):
    transform_train = transforms.Compose([

        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape()
    ])
    transform_test = transforms.Compose([

        transforms.Resize(256),
        transforms.CenterCrop(224),
        HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape()
    ])

    trainset = torchvision.datasets.STL10(root=DATA_DIR, split="train", transform=transform_train, download=True)
    testset = torchvision.datasets.STL10(root=DATA_DIR, split="test", transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloaders(
        train=trainloader, val=testloader,
        test=make_test_loaders(testset, test_shifts, batch_size, num_workers),
    )

def get_camelyon17(n_groups_hue = 1, n_groups_saturation = 1,n_groups_luminance = 1, batch_size=64, frac_space=1.0):
    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader, get_eval_loader

    transform_train = transforms.Compose(
        [
            HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )
    transform_test = transforms.Compose(
        [
            HSVSeparation(n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation,n_groups_luminance=n_groups_luminance,amt=0),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )

    dataset = get_dataset("camelyon17", root_dir=str(DATA_DIR), download=True)
    train_set = dataset.get_subset(
            "train",
            transform=transform_train,
        )
    val_set = dataset.get_subset(
            "val",
            transform=transform_test
        )
    eval_set = dataset.get_subset(
            "test",
            transform=transform_test
        )
    train_loader = get_train_loader(
        "standard",
        train_set,
        batch_size=batch_size)
    val_loader = get_eval_loader(
        "standard",
        val_set,
        batch_size=batch_size)
    eval_loader = get_eval_loader(
        "standard",
        eval_set,
        batch_size=batch_size)

    return dataloaders(train=train_loader, val=val_loader, test=eval_loader)

def get_hue_mnist(
    n_groups_hue=4, n_groups_saturation=1, n_groups_luminance=1, batch_size=128,
    validation_fraction=0.1, split_seed=1999, num_workers=0,
):
    """Hue-shift MNIST with a held-out validation subset of official training data.

    The paper specifies the hue ranges but not the validation fraction. The
    configurable split, uniform hue sampling, fixed saturation/value of one,
    and 0.5 normalization are explicit implementation choices.
    """
    if n_groups_saturation != 1 or n_groups_luminance != 1:
        raise ValueError("Hue-MNIST uses hue lifting only; saturation/luminance orders must be one.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one.")
    transform = transforms.Compose([
        HSVSeparation(n_groups_hue, n_groups_saturation, n_groups_luminance),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
        TensorReshape(),
    ])
    full_train = HSVMNIST(
        DATA_DIR, hue_low=0, hue_high=1 / 3, train=True,
        transform=transform, download=True,
    )
    validation_size = int(len(full_train) * validation_fraction)
    if validation_size == 0 or validation_size == len(full_train):
        raise ValueError("MNIST split must leave at least one training and validation image.")
    trainset, valset = torch.utils.data.random_split(
        full_train, [len(full_train) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
    testset = HSVMNIST(
        DATA_DIR, hue_low=1 / 3, hue_high=1, train=False,
        transform=transform, download=True,
    )

    def loader(dataset, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )

    return dataloaders(train=loader(trainset, True), val=loader(valset), test=loader(testset))


def parse_dataloader(name):
    if name == "shapes3d":
        return get_shapes3d
    elif name == "smallnorb":
        return get_smallnorb
    elif name == "hue_mnist":
        return get_hue_mnist
    elif name == "cifar":
        return get_cifar
    elif name == "cifar100":
        return get_cifar100
    elif name == "caltech101":
        return get_caltech101
    elif name == "camelyon17":
        return get_camelyon17
    elif name == "oxford_pets":
        return get_pets
    elif name == "stanford_cars":
        return get_cars
    elif name == "STL10":
        return get_STL10
    else:
        raise ValueError("Invalid dataset name")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    get_shapes3d(batch_size=4)
