"""
============================================================
  DATA LOADER UTILITY
  -------------------
  Shared helper functions for loading and preparing datasets
  across all four modules (CNN, RNN, LSTM, Combined).

  We use CIFAR-10 throughout the repository because:
    • It auto-downloads (no manual data wrangling)
    • 60 000 images, 10 classes — small enough for laptops
    • 32 × 32 × 3 RGB — real colour images
============================================================
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. STANDARD IMAGE TRANSFORMS
# ============================================================
def get_cifar10_transforms(augment: bool = True):
    """
    Return a pair (train_transform, test_transform) for CIFAR-10.

    Parameters
    ----------
    augment : bool
        If True the training set receives random horizontal flip
        and random crop — standard data-augmentation tricks that
        reduce over-fitting by showing the model slightly
        different versions of each image every epoch.

    Returns
    -------
    (train_transform, test_transform)
    """

    # ----- Mean & Std computed over the CIFAR-10 training set -----
    # These numbers normalise each channel to mean ≈ 0, std ≈ 1
    # so the optimiser converges faster.
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    if augment:
        train_transform = transforms.Compose([
            # Pad 4 px on each side, then randomly crop back to 32×32
            # → effectively shifts the image around.
            transforms.RandomCrop(32, padding=4),

            # 50% chance of horizontal mirror — a car facing left is
            # the same class as a car facing right.
            transforms.RandomHorizontalFlip(),

            # Convert PIL image → float tensor in [0, 1]
            transforms.ToTensor(),

            # Subtract the per-channel mean and divide by std.
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # The test/validation set is NEVER augmented — we want a
    # deterministic evaluation so results are reproducible.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


# ============================================================
# 2. BUILD DATALOADERS
# ============================================================
def get_cifar10_loaders(
    batch_size: int = 64,
    augment: bool = True,
    num_workers: int = 2,
    data_dir: str | None = None,
):
    """
    Download CIFAR-10 (if needed) and return ready-to-use
    DataLoaders for training and testing.

    Parameters
    ----------
    batch_size  : int   – images per mini-batch
    augment     : bool  – use data augmentation for training?
    num_workers : int   – parallel workers for data loading
    data_dir    : str   – where to save/load the dataset

    Returns
    -------
    (train_loader, test_loader, class_names)
    """

    # Default data directory: <project_root>/data
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
        )

    train_transform, test_transform = get_cifar10_transforms(augment)

    # ------ DOWNLOAD & LOAD ------
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,          # 50 000 training images
        download=True,       # auto-download if missing
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,         # 10 000 test images
        download=True,
        transform=test_transform,
    )

    # ------ WRAP IN DATALOADERS ------
    # pin_memory=True speeds up CPU → GPU transfer when a GPU exists.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        # randomise order every epoch
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,       # keep test order fixed for reproducibility
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # CIFAR-10 class labels (index 0 → 'airplane', 1 → 'automobile', …)
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    return train_loader, test_loader, class_names


# ============================================================
# 3. DEVICE HELPER
# ============================================================
def get_device() -> torch.device:
    """
    Return the best available device.

    Priority: CUDA GPU → Apple MPS → CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (training will be slower)")
    return device


# ============================================================
# 4. QUICK SMOKE TEST
# ============================================================
if __name__ == "__main__":
    # Run this file directly to verify data loading works:
    #   python -m src.utils.data_loader
    device = get_device()
    train_loader, test_loader, class_names = get_cifar10_loaders(batch_size=8)

    images, labels = next(iter(train_loader))
    print(f"\nBatch shape : {images.shape}")     # → [8, 3, 32, 32]
    print(f"Labels      : {labels.tolist()}")
    print(f"Class names : {[class_names[l] for l in labels.tolist()]}")
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test  batches: {len(test_loader)}")
    print("\n✅ Data loader is working!")
