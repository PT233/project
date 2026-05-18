"""
classifier.py — EfficientNet-B2 classifier wrapper.

Provides build_classifier(), a factory function that returns an
EfficientNet-B2 model (loaded via timm) with a custom linear head.
"""

import torch
import torch.nn as nn
import timm


def build_classifier(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build an EfficientNet-B2 classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes (replaces the default timm head).
    pretrained : bool
        Whether to load ImageNet-pretrained weights from timm.
    freeze_backbone : bool
        If True, freeze all parameters except the final classifier head
        (feature-extraction mode).

    Returns
    -------
    nn.Module
        EfficientNet-B2 model with a replaced linear classification head.
    """
    model = timm.create_model(
        "efficientnet_b2",
        pretrained=pretrained,
        num_classes=0,  # remove timm's default head; we add our own below
    )

    # Determine the feature dimension produced by the backbone.
    in_features = model.num_features

    # Attach a fresh linear head.
    model.classifier = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        # Freeze every parameter first, then unfreeze the classifier head.
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # DoD verification                                                     #
    # ------------------------------------------------------------------ #

    # DoD ①: instantiate without errors (pretrained=False avoids download)
    model = build_classifier(num_classes=2, pretrained=False)
    print("DoD ①  model instantiated successfully.")

    # DoD ②: forward pass produces shape (2, 2)
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (2, 2), f"Unexpected output shape: {output.shape}"
    print(f"DoD ②  forward pass output shape: {output.shape}  ✓")

    # DoD ③: total parameter count is between 7 M and 10 M
    total_params = sum(p.numel() for p in model.parameters())
    total_m = total_params / 1_000_000
    assert 7 <= total_m <= 10, (
        f"Parameter count {total_m:.2f}M is outside the expected 7–10 M range."
    )
    print(f"DoD ③  total parameters: {total_params:,} ({total_m:.2f} M)  ✓")

    # Extra: verify freeze_backbone works
    frozen_model = build_classifier(num_classes=2, pretrained=False, freeze_backbone=True)
    trainable = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
    print(f"       freeze_backbone check — trainable params: {trainable:,}")

    print("\nAll DoD checks passed.")
