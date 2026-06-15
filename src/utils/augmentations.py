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


def _base_val_transforms() -> list:
    return [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]


def get_transforms(mode: str, profile: str = "breakhis") -> A.Compose:
    """Return an albumentations Compose pipeline for the given mode.

    Args:
        mode: 'train' or 'val'.
        profile: Dataset-specific transform profile. ``'breakhis'`` keeps the
            legacy project pipeline; ``'dlbcl'`` enables stronger H&E-safe
            training augmentation.

    Returns:
        An albumentations.Compose object with the appropriate transforms.

    Raises:
        ValueError: If mode is not 'train' or 'val'.
    """
    profile = str(profile).strip().lower()
    if profile not in {"breakhis", "dlbcl"}:
        raise ValueError(
            f"Unknown transform profile '{profile}'. Expected 'breakhis' or 'dlbcl'."
        )

    if mode == "train":
        if profile == "dlbcl":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=180, p=0.5),
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.05,
                    p=0.7,
                ),
                A.ElasticTransform(alpha=20, sigma=5, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.CoarseDropout(
                    max_holes=4,
                    max_height=32,
                    max_width=32,
                    p=0.3,
                ),
                *_base_val_transforms(),
            ])

        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.ElasticTransform(alpha=20, sigma=5, alpha_affine=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,
                p=0.5,
            ),
            A.CoarseDropout(
                max_holes=4,
                max_height=32,
                max_width=32,
                p=0.3,
            ),
            *_base_val_transforms(),
        ])
    elif mode == "val":
        return A.Compose(_base_val_transforms())
    else:
        raise ValueError(
            f"Unknown transform mode '{mode}'. Expected 'train' or 'val'."
        )
