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

    def __init__(self, csv_path: str, data_root: str, mode: str = "val") -> None:
        self.data_root = data_root
        self.mode = mode
        self.transform = get_transforms(mode)

        self.df = pd.read_csv(csv_path)
        # Validate required columns
        required_cols = {"filepath", "label", "patient_id"}
        missing = required_cols - set(self.df.columns)
        if missing:
            logger.error("[dlbcl_dataset] CSV missing required columns: %s", missing)
            sys.exit(1)

        # Reset index to ensure integer positional access
        self.df = self.df.reset_index(drop=True)

        # Build patient → indices mapping once for fast lookup
        self._patient_to_indices: dict = {}
        for idx, row in self.df.iterrows():
            pid = str(row["patient_id"])
            self._patient_to_indices.setdefault(pid, []).append(int(idx))

    # ------------------------------------------------------------------
    # BaseDataset abstract methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        rel_path = row["filepath"]
        full_path = os.path.join(self.data_root, rel_path) if self.data_root else rel_path

        if not os.path.exists(full_path):
            logger.error("[dlbcl_dataset] Image path does not exist: %s", full_path)
            sys.exit(2)

        image = Image.open(full_path).convert("RGB")
        image_np = np.array(image)

        augmented = self.transform(image=image_np)
        tensor = augmented["image"]  # shape (3, H, W), float32, normalised

        return {
            "image": tensor,
            "label": int(row["label"]),
            "patient_id": str(row["patient_id"]),
            "meta": {},
        }

    # ------------------------------------------------------------------
    # DLBCL-specific helpers
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# __main__ — smoke-test with temporary fake data (no real dataset required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # ── 1. Create temporary fake PNG images ──────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="dlbcl_smoke_")
    rows = []
    patients = ["patient_001", "patient_001", "patient_002", "patient_002", "patient_003"]
    labels   = [1,             1,             0,             0,             1            ]

    for i, (pid, lbl) in enumerate(zip(patients, labels)):
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        fname = f"patch_{i:03d}.png"
        fpath = os.path.join(tmp_dir, fname)
        img.save(fpath)
        rows.append({"filepath": fname, "label": lbl, "patient_id": pid, "magnification": "N/A"})

    csv_path = os.path.join(tmp_dir, "fake_split.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ── 2. Instantiate dataset ────────────────────────────────────────────
    dataset = DLBCLDataset(csv_path=csv_path, data_root=tmp_dir, mode="val")
    logger.info("Dataset length: %d", len(dataset))

    # ── 3. DoD ①: print shape and label ──────────────────────────────────
    sample = dataset[0]
    print(f"Sample shape: {sample['image'].shape}, label: {sample['label']}")
    logger.info("patient_id: %s", sample["patient_id"])

    # ── 4. DoD ②: patient_id consistency ────────────────────────────────
    for pid in dataset.get_patient_ids():
        indices = dataset.get_patches_by_patient(pid)
        pids_for_patient = {dataset[i]["patient_id"] for i in indices}
        assert len(pids_for_patient) == 1, (
            f"patient_id mismatch for {pid}: got {pids_for_patient}"
        )
        assert pid in pids_for_patient, f"Expected {pid}, got {pids_for_patient}"
    logger.info("DoD ②: patient_id consistency verified for all patients.")

    # ── 5. DoD ③: get_patches_by_patient returns ≥1 patch ───────────────
    for pid in dataset.get_patient_ids():
        patches = dataset.get_patches_by_patient(pid)
        assert len(patches) >= 1, f"patient {pid} has no patches!"
        logger.info("patient %s → %d patch(es): indices %s", pid, len(patches), patches)
    logger.info("DoD ③: all patients have ≥1 patch.")

    # ── 6. Cleanup ───────────────────────────────────────────────────────
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("All DoD checks passed.")
