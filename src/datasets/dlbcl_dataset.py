"""
dlbcl_dataset.py — DLBCL patch-level dataset implementation.

Loads DLBCL histology patches from a CSV split file.
Labels are patient-level (all patches for a given patient_id share the same label).

CSV expected columns: filepath, label, patient_id, magnification

__getitem__ return format (protocol.md §2):
    {
        "image":      torch.Tensor,  # shape (3, 224, 224), dtype float32, normalised
        "label":      int,           # 0 = high-survival, 1 = low-survival
        "patient_id": str,           # e.g. "patient_001"
        "meta":       dict           # {} reserved for future use
    }
"""

import logging
import os
import sys
from typing import List

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.datasets.base_dataset import BaseDataset
from src.utils.augmentations import get_transforms

logger = logging.getLogger(__name__)


class DLBCLDataset(BaseDataset):
    """Patch-level DLBCL dataset.

    Parameters
    ----------
    csv_path : str
        Path to the CSV split file with columns:
        filepath, label, patient_id, magnification.
    data_root : str
        Root directory prepended to each ``filepath`` in the CSV
        (use an empty string if filepaths are already absolute).
    mode : str, optional
        Transform mode — ``'train'`` or ``'val'`` (default ``'val'``).
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        mode: str = "val",
        stain_normalization: bool = False,
        stain_reference_path: str = "data/stain_reference.png",
    ) -> None:
        self.csv_path = csv_path
        self.data_root = data_root
        self.mode = mode
        self.transform = get_transforms(mode, profile="dlbcl")
        self.normalizer = None
        self._stain_normalization_enabled = bool(stain_normalization)
        self._stain_reference_path = stain_reference_path

        if self._stain_normalization_enabled:
            self._init_stain_normalizer(stain_reference_path)

        self.df = pd.read_csv(csv_path)
        # Validate required columns
        required_cols = {"filepath", "label", "patient_id"}
        missing = required_cols - set(self.df.columns)
        if missing:
            logger.error("[dlbcl_dataset] CSV missing required columns: %s", missing)
            sys.exit(1)

        self.df["_full_path"] = self.df["filepath"].map(self._resolve_path)
        exists_mask = self.df["_full_path"].map(os.path.exists)
        missing_count = int((~exists_mask).sum())
        if missing_count:
            logger.warning(
                "[dlbcl_dataset] Filtering %d missing image(s) from %s",
                missing_count,
                csv_path,
            )
            self.df = self.df.loc[exists_mask].copy()

        if self.df.empty:
            logger.error("[dlbcl_dataset] No readable images found in CSV: %s", csv_path)
            sys.exit(2)

        # Reset index to ensure integer positional access
        self.df = self.df.reset_index(drop=True)

        # Build patient → indices mapping once for fast lookup
        self._patient_to_indices: dict = {}
        for idx, row in self.df.iterrows():
            pid = str(row["patient_id"])
            self._patient_to_indices.setdefault(pid, []).append(int(idx))

    # BaseDataset abstract methods

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, filepath: str) -> str:
        # CSV split files were generated on a different machine and store
        # absolute paths like "/root/hg_workspace/data/dlbcl/...". Rebase any
        # path onto the configured data_root by anchoring on the "data/" segment
        # so the splits remain portable across machines.
        norm = filepath.replace("\\", "/")
        marker = "/data/"
        if marker in norm:
            rel = norm[norm.index(marker) + 1:]  # keep "data/..."
            if self.data_root:
                return os.path.join(self.data_root, rel)
            return rel
        if os.path.isabs(filepath) or not self.data_root:
            return filepath
        return os.path.join(self.data_root, filepath)

    def _init_stain_normalizer(self, reference_path: str) -> None:
        """Initialise Macenko stain normalisation against a reference patch."""
        if not os.path.exists(reference_path):
            logger.warning(
                "[dlbcl_dataset] Stain reference not found: %s; using raw images",
                reference_path,
            )
            self._stain_normalization_enabled = False
            return

        try:
            import torchstain

            ref_bgr = cv2.imread(reference_path, cv2.IMREAD_COLOR)
            if ref_bgr is None:
                raise ValueError(f"cv2.imread returned None for {reference_path}")

            ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
            normalizer.fit(ref_rgb)
            self.normalizer = normalizer
            logger.info(
                "[dlbcl_dataset] Macenko stain normalization enabled: %s",
                reference_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[dlbcl_dataset] Failed to initialise stain normalizer (%s); using raw images",
                exc,
            )
            self.normalizer = None
            self._stain_normalization_enabled = False

    def _normalize_stain(self, image_np: np.ndarray) -> np.ndarray:
        if self.normalizer is None:
            return image_np
        try:
            normalized, _, _ = self.normalizer.normalize(I=image_np, stains=False)
            return np.clip(normalized, 0, 255).astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[dlbcl_dataset] Stain normalization failed for one patch (%s); using raw image",
                exc,
            )
            return image_np

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        full_path = row["_full_path"]

        image = Image.open(full_path).convert("RGB")
        image_np = np.array(image)
        image_np = self._normalize_stain(image_np)

        augmented = self.transform(image=image_np)
        tensor = augmented["image"]  # shape (3, H, W), float32, normalised

        return {
            "image": tensor,
            "label": int(row["label"]),
            "patient_id": str(row["patient_id"]),
            "meta": {},
        }

    # DLBCL-specific helpers

    def get_patient_ids(self) -> List[str]:
        """Return deduplicated list of patient_id strings."""
        return list(self._patient_to_indices.keys())

    def get_patches_by_patient(self, patient_id: str) -> List[int]:
        """Return list of dataset indices belonging to *patient_id*.

        Parameters
        ----------
        patient_id : str
            Patient identifier as it appears in the CSV.

        Returns
        -------
        List[int]
            Possibly empty list if the patient is not found.
        """
        return list(self._patient_to_indices.get(str(patient_id), []))


# __main__ — smoke-test with temporary fake data (no real dataset required)
