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
    # Force timm to resolve ImageNet weights from the GitHub-release URL
    # (torch.hub cache) instead of the HF Hub, so training works offline
    # when the local cache holds efficientnet_b2_ra-bcdf34b7.pth.
    create_kwargs = dict(pretrained=pretrained, num_classes=0)
    if pretrained:
        create_kwargs["pretrained_cfg_overlay"] = dict(hf_hub_id=None)
    model = timm.create_model("efficientnet_b2", **create_kwargs)

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
