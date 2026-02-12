from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_cifar10_classes() -> list[str]:
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return class_names


def get_dataloaders(
    data_dir: str | Path = "data/",
    batch_size: int = 128,
    num_workers: int = 2,
    valid_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    mean: tuple[float, float, float] = (0.4919, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

    dataset_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    training_data = CIFAR10(root=data_dir, train=True, download=True, transform=dataset_transforms)
    test_set = CIFAR10(root=data_dir, train=False, download=True, transform=dataset_transforms)

    training_size = len(training_data)
    validation_length = int(training_size * valid_split)
    training_length = int(training_size - validation_length)

    g = torch.Generator().manual_seed(seed)
    training_set, validation_set = random_split(training_data, [training_length, validation_length], generator=g)

    train = DataLoader(
        training_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    validation = DataLoader(
        validation_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    test = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train, validation, test
