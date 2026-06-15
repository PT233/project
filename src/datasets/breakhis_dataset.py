"""
breakhis_dataset.py — BreaKHis histology dataset loader.

Loads 40x magnification slides from a CSV split file and returns
standardised sample dicts (protocol.md §2).

CSV expected columns: filepath, label, patient_id, magnification
Only rows where magnification == '40x' are loaded.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

from src.datasets.base_dataset import BaseDataset
from src.utils.augmentations import get_transforms

logger = logging.getLogger(__name__)


class BreaKHisDataset(BaseDataset):
    """BreaKHis binary classification dataset (benign vs. malignant).

    Only 40x magnification images are loaded, as specified by protocol.md.

    Parameters
    ----------
    csv_path : str
        Path to the CSV split file
        (columns: filepath, label, patient_id, magnification).
    data_root : str
        Root directory of the dataset; ``filepath`` values in the CSV are
        treated as relative to the *project root*, not to ``data_root``.
        ``data_root`` is kept for future use / compatibility.
    mode : str
        Transform mode: ``'train'`` or ``'val'`` (default ``'val'``).
    """

    MAGNIFICATION_FILTER = "40x"

    def __init__(self, csv_path: str, data_root: str, mode: str = "val",
                 augmentation: bool = True) -> None:
        self.csv_path = csv_path
        self.data_root = data_root
        self.mode = mode
        # When augmentation=False, always use val transforms (Resize+Normalize only)
        effective_mode = mode if augmentation else "val"
        self.transform = get_transforms(effective_mode)

        df = pd.read_csv(csv_path)
        df = df[df["magnification"] == self.MAGNIFICATION_FILTER].reset_index(drop=True)

        self.filepaths = df["filepath"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.patient_ids = df["patient_id"].astype(str).tolist()

        logger.info(
            "[BreaKHisDataset] Loaded %d samples (magnification=%s, mode=%s)",
            len(self.filepaths),
            self.MAGNIFICATION_FILTER,
            mode,
        )

    # BaseDataset interface

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> dict:
        filepath = self.filepaths[index]

        if not os.path.exists(filepath):
            logger.error("[BreaKHisDataset] Image path does not exist: %s", filepath)
            sys.exit(2)

        image = Image.open(filepath).convert("RGB")
        image_np = np.array(image)

        augmented = self.transform(image=image_np)
        image_tensor = augmented["image"]  # shape: (3, 224, 224), float32

        return {
            "image": image_tensor,
            "label": self.labels[index],
            "patient_id": self.patient_ids[index],
            "meta": {},
        }


# Smoke-test: run with `python src/datasets/breakhis_dataset.py`
# Uses temporary fake data — no real images required.
