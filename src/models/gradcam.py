"""
gradcam.py — GradCAM heat-map generation for the EfficientNet-B2 classifier.

Usage
-----
python src/models/gradcam.py --image <path> --checkpoint <ckpt_path> [--output <dir>]

The script loads a checkpoint produced by train.py, runs GradCAM on the given
image using the last point-wise-linear conv of the final EfficientNet-B2 block
(``model.blocks[-1][-1].conv_pwl``) as the target layer, and writes an
overlay PNG to the output directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── local import ──────────────────────────────────────────────────────────── #
# Allow running the script from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.classifier import build_classifier  # noqa: E402

# ── logging ───────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s][gradcam] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gradcam")

# ── constants ─────────────────────────────────────────────────────────────── #
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.224, 0.225, 0.229)

_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


# ── helpers ───────────────────────────────────────────────────────────────── #

def _load_model(checkpoint_path: str) -> torch.nn.Module:
    """Load an EfficientNet-B2 classifier from a checkpoint file.

    The checkpoint must follow the protocol defined in protocol.md §2:
    ``{"epoch": int, "model_state": OrderedDict, "optim_state": OrderedDict,
       "best_auc": float, "config": dict}``

    Only ``model_state`` is consumed here; all other keys are ignored.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" not in ckpt:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' does not contain key 'model_state'. "
            f"Found keys: {list(ckpt.keys())}"
        )

    cfg = ckpt.get("config", {})
    num_classes = cfg.get("num_classes", 2)

    model = build_classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _preprocess_image(image_path: str) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess an image for the model.

    Returns
    -------
    tensor : torch.Tensor
        Preprocessed image tensor of shape (1, 3, 224, 224).
    rgb_float : np.ndarray
        Resized image as float32 in [0, 1] for overlay rendering, shape
        (224, 224, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224), Image.BILINEAR)
    rgb_float = np.array(img_resized, dtype=np.float32) / 255.0

    tensor = _VAL_TRANSFORM(img).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor, rgb_float


def generate_gradcam(
    image_path: str,
    checkpoint_path: str,
    output_dir: str = "artifacts/results/gradcam/",
) -> Path:
    """Generate a GradCAM overlay for *image_path* and save it to *output_dir*.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    checkpoint_path : str
        Path to the model checkpoint (see protocol.md §2 for format).
    output_dir : str
        Directory where the output PNG will be written.

    Returns
    -------
    Path
        Absolute path to the saved overlay PNG.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from '%s'", checkpoint_path)
    model = _load_model(checkpoint_path)

    # Target layer: last point-wise-linear conv in the last EfficientNet-B2 block.
    target_layer = model.blocks[-1][-1].conv_pwl
    logger.info("Target layer: model.blocks[-1][-1].conv_pwl  (%s)", target_layer)

    logger.info("Preprocessing image '%s'", image_path)
    input_tensor, rgb_float = _preprocess_image(image_path)

    # GradCAM expects the model to return logits; class_idx=None uses the
    # argmax of the forward pass (highest-score class).
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)  # (1, 224, 224)
    grayscale_cam = grayscale_cam[0]  # (224, 224)

    # Overlay heat-map on the original image.
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)  # uint8

    stem = Path(image_path).stem
    out_path = out_dir / f"{stem}_gradcam.png"
    Image.fromarray(overlay).save(out_path)
    logger.info("Saved GradCAM overlay to '%s'", out_path)

    return out_path.resolve()


# ── CLI ───────────────────────────────────────────────────────────────────── #

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a GradCAM heat-map overlay for a single image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output",
        default="artifacts/results/gradcam/",
        help="Output directory (default: artifacts/results/gradcam/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if not os.path.isfile(args.image):
        logger.error("Image file not found: '%s'", args.image)
        sys.exit(2)
    if not os.path.isfile(args.checkpoint):
        logger.error("Checkpoint file not found: '%s'", args.checkpoint)
        sys.exit(2)

    out_path = generate_gradcam(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
    )
    print(f"GradCAM saved to: {out_path}")


# ── spec self-test ─────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    main()
