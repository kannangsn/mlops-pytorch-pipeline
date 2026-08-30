"""Data loading for CIFAR-10 (plus a fake dataset for quick tests/CI)."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Channel statistics computed over the CIFAR-10 training split.
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(train: bool = True) -> transforms.Compose:
    """Build the image transform pipeline: augmented for training, plain for eval."""
    normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    if train:
        # Light augmentation (flip + random crop) to reduce overfitting.
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
    # No augmentation at eval time - just tensor conversion and normalization.
    return transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


def _build_datasets(dataset: str, data_dir: str):
    """Construct the (train, val) dataset pair for the requested dataset name."""
    if dataset == "cifar10":
        train_ds = datasets.CIFAR10(
            root=data_dir, train=True, download=True,
            transform=get_transforms(train=True),
        )
        val_ds = datasets.CIFAR10(
            root=data_dir, train=False, download=True,
            transform=get_transforms(train=False),
        )
    elif dataset == "fake":
        # Random images with the same shape as CIFAR-10. Used by the unit
        # tests and CI smoke run so they don't need the 170MB download.
        train_ds = datasets.FakeData(
            size=256, image_size=(3, 32, 32), num_classes=10,
            transform=get_transforms(train=True),
        )
        val_ds = datasets.FakeData(
            size=64, image_size=(3, 32, 32), num_classes=10,
            transform=get_transforms(train=False),
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: cifar10, fake")
    return train_ds, val_ds


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    dataset: str = "cifar10",
) -> tuple[DataLoader, DataLoader]:
    """Build the train/val DataLoaders (shuffled for train, sequential for val)."""
    train_ds, val_ds = _build_datasets(dataset, data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
