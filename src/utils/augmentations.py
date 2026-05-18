"""
augmentations.py — Train/Val transform definitions using albumentations.

Usage:
    from src.utils.augmentations import get_transforms
    transform = get_transforms('train')  # or 'val'
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization constants (protocol.md §3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Target image size (protocol.md §2)
IMAGE_SIZE = 224


def get_transforms(mode: str) -> A.Compose:
    """Return an albumentations Compose pipeline for the given mode.

    Args:
        mode: 'train' or 'val'.

    Returns:
        An albumentations.Compose object with the appropriate transforms.

    Raises:
        ValueError: If mode is not 'train' or 'val'.
    """
    if mode == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    elif mode == "val":
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        raise ValueError(
            f"Unknown transform mode '{mode}'. Expected 'train' or 'val'."
        )


if __name__ == "__main__":
    print("=== train transforms ===")
    print(get_transforms("train"))
    print()
    print("=== val transforms ===")
    print(get_transforms("val"))
