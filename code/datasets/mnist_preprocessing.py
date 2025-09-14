import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from collections import Counter


def create_balanced_sampler(dataset, indices):
    """Create a weighted sampler to balance classes for a subset"""
    targets = [dataset[i][1] for i in indices]
    class_counts = Counter(targets)

    # calculate weights for each class (inverse frequency)
    total_samples = len(targets)
    num_classes = len(class_counts)

    class_weights = {}
    for class_id, count in class_counts.items():
        class_weights[class_id] = total_samples / (num_classes * count)

    sample_weights = [class_weights[target] for target in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    print(f"Class distribution: {dict(class_counts)}")
    print(f"Class weights: {class_weights}")

    return sampler


def get_advanced_transforms():
    """Create data augmentations"""

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5,
                fill=0,
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3, fill=0),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(
                lambda x: x + torch.randn_like(x) * 0.01
            ),  # add small noise
        ]
    )

    # validation/test transforms (minimal augmentation)
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    return train_transform, val_transform


def preprocess_dataset():
    """Preprocess MNIST dataset with advanced techniques"""

    print("Creating data transforms...")
    train_transform, val_transform = get_advanced_transforms()

    print("Loading MNIST dataset...")
    train_dataset = torchvision.datasets.MNIST(
        root="../../data", train=True, download=True, transform=train_transform
    )

    val_dataset = torchvision.datasets.MNIST(
        root="../../data", train=False, download=True, transform=val_transform
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    print("Creating balanced class sampler...")
    train_sampler = create_balanced_sampler(train_dataset, train_subset.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=128,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_subset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False
    )

    test_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False
    )

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(val_dataset)}")

    return train_loader, val_loader, test_loader
