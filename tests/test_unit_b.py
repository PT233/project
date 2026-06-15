"""
test_unit_b.py — Unit tests for classifier.py and gradcam.py
Covers UT-10 through UT-14 as specified in test.md §1.4 and §1.5.
"""

import sys
import tempfile
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

# Ensure project root is on sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classifier import build_classifier  # noqa: E402
from src.models.gradcam import generate_gradcam      # noqa: E402


# Fixtures

@pytest.fixture(scope="module")
def model_2class():
    """Build a 2-class classifier once for the whole module."""
    return build_classifier(num_classes=2, pretrained=False)


@pytest.fixture(scope="module")
def fake_checkpoint_and_image(tmp_path_factory):
    """
    Create a temporary directory containing:
      - fake.pth  : a valid checkpoint using build_classifier weights
      - image.png : a random 300x400 RGB image
    Returns (ckpt_path, img_path, out_dir) as strings.
    """
    tmp = tmp_path_factory.mktemp("gradcam_test")

    # Build fake checkpoint
    ref_model = build_classifier(num_classes=2, pretrained=False)
    ckpt = {
        "epoch": 0,
        "model_state": OrderedDict(ref_model.state_dict()),
        "optim_state": OrderedDict(),
        "best_auc": 0.5,
        "config": {"num_classes": 2},
    }
    ckpt_path = tmp / "fake.pth"
    torch.save(ckpt, ckpt_path)

    # Build fake image with pixel values in (30, 220) to guarantee non-trivial mean
    rng = np.random.default_rng(42)
    img_array = rng.integers(30, 220, size=(300, 400, 3), dtype=np.uint8)
    img_path = tmp / "test_sample.png"
    Image.fromarray(img_array).save(img_path)

    out_dir = tmp / "gradcam_out"

    return str(ckpt_path), str(img_path), str(out_dir)


# UT-10: build_classifier returns nn.Module with correct output features

class TestUT10:
    def test_returns_nn_module(self):
        """UT-10: build_classifier returns an nn.Module instance."""
        model = build_classifier(num_classes=2, pretrained=False)
        assert isinstance(model, nn.Module), (
            f"Expected nn.Module, got {type(model)}"
        )

    def test_classifier_out_features(self):
        """UT-10: The classifier head has out_features == 2."""
        model = build_classifier(num_classes=2, pretrained=False)
        head = model.classifier
        # head is nn.Linear directly
        if isinstance(head, nn.Linear):
            out_features = head.out_features
        elif hasattr(head, "__getitem__"):
            # head is a Sequential or list-like; take the last element
            out_features = head[-1].out_features
        else:
            pytest.fail(f"Cannot determine out_features from classifier type: {type(head)}")
        assert out_features == 2, (
            f"Expected out_features=2, got {out_features}"
        )


# UT-11: forward pass produces shape (2, 2)

class TestUT11:
    def test_forward_output_shape(self, model_2class):
        """UT-11: model(torch.randn(2, 3, 224, 224)) output shape == (2, 2)."""
        model_2class.eval()
        with torch.no_grad():
            output = model_2class(torch.randn(2, 3, 224, 224))
        assert output.shape == (2, 2), (
            f"Expected shape (2, 2), got {tuple(output.shape)}"
        )


# UT-12: softmax probabilities sum to 1.0 (per sample)

class TestUT12:
    def test_softmax_sums_to_one(self, model_2class):
        """UT-12: softmax output rows sum to 1.0 within 1e-5."""
        model_2class.eval()
        with torch.no_grad():
            logits = model_2class(torch.randn(1, 3, 224, 224))
        probs = torch.softmax(logits, dim=1)  # shape (1, 2)
        row_sum = probs.sum(dim=1).item()
        assert abs(row_sum - 1.0) < 1e-5, (
            f"Softmax row sum expected 1.0, got {row_sum}"
        )


# UT-13: generate_gradcam returns an ndarray with shape (H, W, 3)

class TestUT13:
    def test_gradcam_returns_overlay_png(self, fake_checkpoint_and_image):
        """UT-13: generate_gradcam saves a PNG and returns a valid path."""
        ckpt_path, img_path, out_dir = fake_checkpoint_and_image
        result_path = generate_gradcam(
            image_path=img_path,
            checkpoint_path=ckpt_path,
            output_dir=out_dir,
        )
        assert result_path.exists(), f"Output PNG not found: {result_path}"

    def test_gradcam_output_is_rgb_array(self, fake_checkpoint_and_image):
        """UT-13: The saved PNG can be loaded as an ndarray with shape (H, W, 3)."""
        ckpt_path, img_path, out_dir = fake_checkpoint_and_image
        result_path = generate_gradcam(
            image_path=img_path,
            checkpoint_path=ckpt_path,
            output_dir=out_dir,
        )
        overlay = np.array(Image.open(result_path))
        assert overlay.ndim == 3, f"Expected 3D array, got {overlay.ndim}D"
        assert overlay.shape[2] == 3, (
            f"Expected 3 channels (RGB), got {overlay.shape[2]}"
        )


# UT-14: overlay pixel mean is in (10, 245)

class TestUT14:
    def test_gradcam_pixel_mean_in_range(self, fake_checkpoint_and_image):
        """UT-14: GradCAM overlay pixel mean > 10 and < 245."""
        ckpt_path, img_path, out_dir = fake_checkpoint_and_image
        result_path = generate_gradcam(
            image_path=img_path,
            checkpoint_path=ckpt_path,
            output_dir=out_dir,
        )
        overlay = np.array(Image.open(result_path), dtype=np.float32)
        mean_val = float(overlay.mean())
        assert mean_val > 10, (
            f"Pixel mean {mean_val:.2f} <= 10 (image may be all-black)"
        )
        assert mean_val < 245, (
            f"Pixel mean {mean_val:.2f} >= 245 (image may be all-white)"
        )
