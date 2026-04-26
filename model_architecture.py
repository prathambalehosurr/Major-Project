"""
model_architecture.py
─────────────────────
Dual-branch architecture for brain-tumour classification:
  Branch A : ResNet50  + CBAM attention  (your original branch)
  Branch B : ConvNeXt-Small              (new, pre-trained)
  Fusion   : Feature-level concatenation → shared MLP head

Design rationale
────────────────
* Feature-level fusion (rather than late/prediction-level ensemble) lets the
  shared classification head learn cross-branch interactions, which typically
  outperforms simple averaging while still using two diverse inductive biases.
* CBAM on ResNet gives explicit spatial + channel attention on CNN features.
* ConvNeXt provides a modern convolution hierarchy with better scaling behaviour
  and stronger ImageNet priors, complementing ResNet's residual features.
* A single, jointly-trained head means only ONE set of classification weights
  to tune, keeping parameter count manageable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    ConvNeXt_Small_Weights,
    resnet50,
    convnext_small,
)


# ──────────────────────────────────────────────
#  Helpers / parameter filters (kept from orig)
# ──────────────────────────────────────────────

def is_classifier_or_attention_parameter(name: str) -> bool:
    """Return True for parameters that should always be trained."""
    keywords = ("fc", "classifier", "cbam", "fusion_head")
    return any(kw in name for kw in keywords)


# ──────────────────────────────────────────────
#  CBAM  (Channel + Spatial Attention)
# ──────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *_ = x.shape
        avg = x.mean(dim=(2, 3))
        mx  = x.amax(dim=(2, 3))
        scale = torch.sigmoid(
            self.shared_mlp(avg) + self.shared_mlp(mx)
        ).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.bn   = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        scale = torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# ──────────────────────────────────────────────
#  Branch A : ResNet50 + CBAM
# ──────────────────────────────────────────────

class ResNet50WithCBAM(nn.Module):
    """
    ResNet50 backbone with CBAM inserted after layer4, before global avg pool.
    FC head removed — exposes raw 2048-d features.
    """

    def __init__(self, pretrained: bool = True, cbam_reduction: int = 16):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)

        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.cbam    = CBAM(2048, reduction=cbam_reduction)
        self.avgpool = base.avgpool
        self.feature_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)          # (B, 2048)

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if "cbam" not in name:
                p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad_(True)


# ──────────────────────────────────────────────
#  Branch B : ConvNeXt-Small
# ──────────────────────────────────────────────

class ConvNeXtBranch(nn.Module):
    """
    ConvNeXt-Small backbone with classification head removed.
    Exposes 768-d feature vectors.

    IMPORTANT: self.norm (LayerNorm) must be applied BEFORE torch.flatten
    because it expects a 4D input (B, C, H, W), not a 2D input (B, C).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        base = convnext_small(weights=weights)

        self.features    = base.features        # all conv stages
        self.avgpool     = base.avgpool         # AdaptiveAvgPool2d(1,1)
        self.norm        = base.classifier[0]   # LayerNorm — expects 4D
        self.feature_dim = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)                     # (B, 768, 1, 1) — still 4D
        x = self.norm(x)                        # LayerNorm needs 4D input
        x = torch.flatten(x, 1)                 # (B, 768) — flatten last
        return x

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad_(True)


# ──────────────────────────────────────────────
#  Fusion Head MLP
# ──────────────────────────────────────────────

class FusionHead(nn.Module):
    """
    Two-layer MLP: concatenated branch features → class logits.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int,
                 dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
#  Dual-Branch Fusion Model  (top-level)
# ──────────────────────────────────────────────

class DualBranchTumorClassifier(nn.Module):
    """
    ResNet50+CBAM  ──┐
                     ├─ concat(2048+768=2816) ─► FusionHead ─► logits
    ConvNeXt-Small ──┘

    Args
    ────
    num_classes   : number of output classes (3 for Meningioma/Glioma/Pituitary)
    pretrained    : load ImageNet weights for both branches
    hidden_dim    : width of the intermediate layer in FusionHead
    dropout       : dropout rate in FusionHead
    cbam_reduction: reduction ratio inside CBAM channel attention
    """

    def __init__(
        self,
        num_classes:    int   = 3,
        pretrained:     bool  = True,
        hidden_dim:     int   = 512,
        dropout:        float = 0.4,
        cbam_reduction: int   = 16,
    ):
        super().__init__()
        self.branch_resnet  = ResNet50WithCBAM(pretrained, cbam_reduction)
        self.branch_convnxt = ConvNeXtBranch(pretrained)

        fused_dim = self.branch_resnet.feature_dim + self.branch_convnxt.feature_dim
        self.fusion_head = FusionHead(fused_dim, hidden_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_r = self.branch_resnet(x)              # (B, 2048)
        feat_c = self.branch_convnxt(x)             # (B, 768)
        fused  = torch.cat([feat_r, feat_c], dim=1) # (B, 2816)
        return self.fusion_head(fused)              # (B, num_classes)

    # ── freeze / unfreeze helpers ───────────────

    def freeze_backbones(self):
        self.branch_resnet.freeze_backbone()
        self.branch_convnxt.freeze_backbone()
        print("[freeze] Both backbones frozen — only FusionHead trains.")

    def unfreeze_backbones(self):
        self.branch_resnet.unfreeze_backbone()
        self.branch_convnxt.unfreeze_backbone()
        print("[unfreeze] Both backbones unfrozen — full fine-tuning active.")

    def unfreeze_resnet_only(self):
        self.branch_resnet.unfreeze_backbone()
        print("[unfreeze] ResNet50+CBAM branch unfrozen.")

    def unfreeze_convnext_only(self):
        self.branch_convnxt.unfreeze_backbone()
        print("[unfreeze] ConvNeXt branch unfrozen.")

    def parameter_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters — total: {total:,}  |  trainable: {trainable:,}")

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"[save] Model weights saved → {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> "DualBranchTumorClassifier":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"[load] Model weights loaded ← {path}")
        return model


# ──────────────────────────────────────────────
#  Convenience builders
# ──────────────────────────────────────────────

def build_resnet50_model(num_classes: int = 3) -> ResNet50WithCBAM:
    """Kept for backward-compatibility."""
    return ResNet50WithCBAM(pretrained=True)


def build_dual_branch_model(
    num_classes:    int   = 3,
    pretrained:     bool  = True,
    hidden_dim:     int   = 512,
    dropout:        float = 0.4,
    cbam_reduction: int   = 16,
) -> DualBranchTumorClassifier:
    return DualBranchTumorClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        dropout=dropout,
        cbam_reduction=cbam_reduction,
    )