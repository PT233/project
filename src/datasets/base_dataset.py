"""
base_dataset.py — Abstract base class for all dataset implementations.

Subclass Contract
-----------------
Every concrete subclass **must** implement both abstract methods and ensure
that ``__getitem__`` returns a dict with exactly the following structure
(field names and types are fixed by protocol.md §2):

    {
        "image":      torch.Tensor,  # shape (3, 224, 224), dtype float32, normalised
        "label":      int,           # 0 = benign/high-survival, 1 = malignant/low-survival
        "patient_id": str,           # e.g. "SOB_B_A-14-22549G" or "patient_001"
        "meta":       dict           # currently always {}, reserved for future use
    }

Violating the contract (wrong keys, wrong tensor shape, wrong dtype) will
cause silent failures downstream in training and evaluation pipelines.
"""

from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Abstract base class for cancer histology datasets.

    All dataset implementations (BreaKHis, DLBCL, …) must inherit from this
    class and implement :meth:`__len__` and :meth:`__getitem__`.

    Return format of ``__getitem__`` (protocol.md §2 — **do not deviate**)::

        {
            "image":      torch.Tensor,  # shape (3, 224, 224), dtype float32
            "label":      int,           # 0 or 1
            "patient_id": str,
            "meta":       dict           # {} unless extended by subclass
        }
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Return a single sample dict for *index*.

        Parameters
        ----------
        index:
            Integer index in ``[0, len(self))``.

        Returns
        -------
        dict with keys ``"image"``, ``"label"``, ``"patient_id"``, ``"meta"``.
        See module docstring for the exact field specification.
        """
