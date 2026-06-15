"""
test_unit_a.py — Unit tests for augmentations.py, logger.py, base_dataset.py
Test Subagent A: UT-01 to UT-09
"""

import os
import subprocess
import sys

import numpy as np
import pytest

# Ensure the project root is on the path so imports work
sys.path.insert(0, "/mnt/h/1.MasterDegree/17.RD/project")

import albumentations as A

from src.utils.augmentations import get_transforms

# UT-01 to UT-04: augmentations.py


class TestGetTransforms:
    """UT-01 to UT-04: tests for get_transforms()"""

    def test_ut01_train_contains_required_transforms(self):
        """UT-01: get_transforms('train') contains HorizontalFlip, Rotate, ColorJitter."""
        transform = get_transforms("train")
        assert isinstance(transform, A.Compose), "Should return A.Compose"
        transform_types = [type(t).__name__ for t in transform.transforms]
        assert "HorizontalFlip" in transform_types, "HorizontalFlip missing from train transforms"
        assert "Rotate" in transform_types, "Rotate missing from train transforms"
        assert "ColorJitter" in transform_types, "ColorJitter missing from train transforms"

    def test_ut02_val_only_resize_and_normalize(self):
        """UT-02: get_transforms('val') only contains Resize + Normalize, no Flip/Rotation/Jitter."""
        transform = get_transforms("val")
        assert isinstance(transform, A.Compose), "Should return A.Compose"
        transform_types = [type(t).__name__ for t in transform.transforms]
        # Must NOT contain augmentation-only transforms
        assert "HorizontalFlip" not in transform_types, "HorizontalFlip should not be in val transforms"
        assert "Rotate" not in transform_types, "Rotate should not be in val transforms"
        assert "ColorJitter" not in transform_types, "ColorJitter should not be in val transforms"
        # Must contain Resize and Normalize
        assert "Resize" in transform_types, "Resize should be in val transforms"
        assert "Normalize" in transform_types, "Normalize should be in val transforms"

    def test_ut03_invalid_mode_raises_value_error(self):
        """UT-03: get_transforms('invalid') raises ValueError."""
        with pytest.raises(ValueError):
            get_transforms("invalid")

    def test_ut04_train_transform_output_shape_and_range(self):
        """UT-04: random numpy uint8 array (300,300,3) -> tensor shape (3,224,224), values in [-3,3]."""
        import torch

        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8)

        transform = get_transforms("train")
        result = transform(image=image)
        tensor = result["image"]

        assert isinstance(tensor, torch.Tensor), "Output should be a torch.Tensor"
        assert tensor.shape == (3, 224, 224), f"Expected shape (3,224,224), got {tensor.shape}"
        assert tensor.min().item() >= -3.0, f"Min value {tensor.min().item()} is below -3"
        assert tensor.max().item() <= 3.0, f"Max value {tensor.max().item()} is above 3"


# UT-05 to UT-07: logger.py


class TestWandbLogger:
    """UT-05 to UT-07: tests for WandbLogger."""

    def test_ut05_init_with_api_key_succeeds(self, monkeypatch):
        """UT-05: WandbLogger initializes successfully when WANDB_API_KEY is set (offline mode)."""
        monkeypatch.setenv("WANDB_API_KEY", "fake_key_for_testing_1234567890abcdef")
        monkeypatch.setenv("WANDB_MODE", "offline")

        from src.utils.logger import WandbLogger

        logger = WandbLogger(config={"lr": 1e-4, "epochs": 1}, project="test-project")
        assert logger is not None
        logger.finish()

    def test_ut06_missing_api_key_exits_with_code_1(self):
        """UT-06: WandbLogger() with no WANDB_API_KEY exits with code 1 (subprocess isolated)."""
        env = os.environ.copy()
        env.pop("WANDB_API_KEY", None)  # ensure key is absent

        code = (
            "import sys; sys.path.insert(0, '/mnt/h/1.MasterDegree/17.RD/project'); "
            "from src.utils.logger import WandbLogger; "
            "WandbLogger(config={}, project='test')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, (
            f"Expected exit code 1, got {result.returncode}. "
            f"stderr: {result.stderr}"
        )

    def test_ut07_log_does_not_raise(self, monkeypatch):
        """UT-07: logger.log({'loss': 0.5}) does not raise any exception."""
        monkeypatch.setenv("WANDB_API_KEY", "fake_key_for_testing_1234567890abcdef")
        monkeypatch.setenv("WANDB_MODE", "offline")

        from src.utils.logger import WandbLogger

        logger = WandbLogger(config={"lr": 1e-4, "epochs": 1}, project="test-project")
        # Should not raise
        logger.log({"loss": 0.5})
        logger.finish()


# UT-08 to UT-09: base_dataset.py


class TestBaseDataset:
    """UT-08 to UT-09: tests for BaseDataset."""

    def test_ut08_direct_instantiation_raises_type_error(self):
        """UT-08: BaseDataset() raises TypeError (abstract class)."""
        from src.datasets.base_dataset import BaseDataset

        with pytest.raises(TypeError):
            BaseDataset()

    def test_ut09_subclass_without_getitem_raises_type_error(self):
        """UT-09: Subclass implementing only __len__ raises TypeError on instantiation or __getitem__ call."""
        from src.datasets.base_dataset import BaseDataset

        # A subclass that only implements __len__ but not __getitem__
        class IncompleteDataset(BaseDataset):
            def __len__(self):
                return 5

        # ABC requires both abstract methods to be implemented;
        # Python raises TypeError when trying to instantiate a class with
        # unimplemented abstract methods.
        with pytest.raises(TypeError):
            dataset = IncompleteDataset()
            # If somehow instantiation succeeds, calling __getitem__ should fail
            dataset[0]
