"""ABMIL: EfficientNet-B2 + gated attention MIL for DLBCL (end-to-end, bag-level)."""

import torch
import torch.nn as nn
import timm


class GatedAttention(nn.Module):
    """Gated attention pooling over instances (Ilse et al., ICML 2018)."""

    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.attn_v = nn.Linear(in_dim, hidden_dim)
        self.attn_u = nn.Linear(in_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, feats):
        v = torch.tanh(self.attn_v(feats))
        u = torch.sigmoid(self.attn_u(feats))
        scores = self.attn_w(v * u).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        bag_feat = torch.bmm(weights.unsqueeze(1), feats).squeeze(1)
        return bag_feat, weights


class ABMIL(nn.Module):
    """EfficientNet-B2 + gated attention MIL classifier."""

    def __init__(self, num_classes=2, pretrained=True, backbone="efficientnet_b2",
                 attn_hidden=128, dropout=0.0):
        super().__init__()
        create_kwargs = dict(pretrained=pretrained, num_classes=0)
        if pretrained:
            create_kwargs["pretrained_cfg_overlay"] = dict(hf_hub_id=None)
        self.backbone = timm.create_model(backbone, **create_kwargs)
        feat_dim = self.backbone.num_features
        self.attention = GatedAttention(feat_dim, attn_hidden)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, bags):
        b, n, c, h, w = bags.shape
        x = bags.reshape(b * n, c, h, w)
        feats = self.backbone(x).reshape(b, n, -1)
        bag_feat, weights = self.attention(feats)
        bag_feat = self.dropout(bag_feat)
        logits = self.head(bag_feat)
        return logits, weights


def build_mil_model(num_classes=2, pretrained=True, attn_hidden=128, dropout=0.0):
    return ABMIL(num_classes=num_classes, pretrained=pretrained,
                 attn_hidden=attn_hidden, dropout=dropout)
