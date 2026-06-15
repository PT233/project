"""DLBCLBagDataset: patient-level bag dataset for ABMIL (end-to-end MIL).

Wraps the patch-level DLBCLDataset. Each item is one patient's "bag":
a stack of N sampled patch tensors plus the patient-level label.

Item:
    {
        "bag":        FloatTensor (N, 3, 224, 224),
        "label":      int,
        "patient_id": str,
        "num_patches": int,   # patches actually available for this patient
    }

Sampling (train): random N patches with replacement if the patient has < N.
Sampling (val/eval): deterministic — first N (or all, tiled) for reproducibility.
"""

import numpy as np
import torch

from src.datasets.dlbcl_dataset import DLBCLDataset


class DLBCLBagDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, data_root, mode="val",
                 bag_size=64, stain_normalization=False,
                 stain_reference_path="data/stain_reference.png", seed=42):
        self.patch_ds = DLBCLDataset(
            csv_path=csv_path,
            data_root=data_root,
            mode=mode,
            stain_normalization=stain_normalization,
            stain_reference_path=stain_reference_path,
        )
        self.bag_size = int(bag_size)
        self.mode = mode
        self._rng = np.random.default_rng(seed)
        self.patient_ids = self.patch_ds.get_patient_ids()
        # patient label = label of its first patch (labels are patient-consistent)
        self._labels = {}
        for pid in self.patient_ids:
            idxs = self.patch_ds.get_patches_by_patient(pid)
            self._labels[pid] = int(self.patch_ds.df.iloc[idxs[0]]["label"])

    def __len__(self):
        return len(self.patient_ids)

    def _select_indices(self, idxs):
        n = len(idxs)
        k = self.bag_size
        if self.mode == "train":
            replace = n < k
            chosen = self._rng.choice(idxs, size=k, replace=replace)
        else:
            # deterministic: tile/truncate to exactly k
            reps = int(np.ceil(k / n))
            chosen = (idxs * reps)[:k]
        return list(chosen)

    def __getitem__(self, index):
        pid = self.patient_ids[index]
        idxs = self.patch_ds.get_patches_by_patient(pid)
        chosen = self._select_indices(idxs)
        patches = [self.patch_ds[i]["image"] for i in chosen]
        bag = torch.stack(patches, dim=0)  # (N, 3, 224, 224)
        return {
            "bag": bag,
            "label": self._labels[pid],
            "patient_id": str(pid),
            "num_patches": len(idxs),
        }
