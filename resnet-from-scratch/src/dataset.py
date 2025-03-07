"""
CIFAR-10 data pipeline with augmentations.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10(batch_size: int = 128, data_root: str = "./data",
                num_workers: int = 2) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and test DataLoaders.

    Training augmentations follow the ResNet paper:
    - 4-pixel padding + random 32x32 crop
    - Random horizontal flip
    - Per-channel normalization
    """

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader
