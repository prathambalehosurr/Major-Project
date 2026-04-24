import torch
from torch import nn
from torchvision.models import resnet50


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_attention = self.shared_mlp(self.avg_pool(x))
        max_attention = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg_attention + max_attention)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attention


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        return self.spatial_attention(x)


def classifier_head(in_features, num_classes=4):
    return nn.Sequential(
        nn.Linear(in_features, 2048),
        nn.SELU(),
        nn.Dropout(p=0.4),
        nn.Linear(2048, 2048),
        nn.SELU(),
        nn.Dropout(p=0.4),
        nn.Linear(2048, num_classes),
    )


def add_cbam_to_resnet(model):
    model.layer1 = nn.Sequential(model.layer1, CBAMBlock(256))
    model.layer2 = nn.Sequential(model.layer2, CBAMBlock(512))
    model.layer3 = nn.Sequential(model.layer3, CBAMBlock(1024))
    model.layer4 = nn.Sequential(model.layer4, CBAMBlock(2048))
    return model


def build_resnet50_model(weights=None, attention="cbam", num_classes=4):
    model = resnet50(weights=weights)
    if attention == "cbam":
        model = add_cbam_to_resnet(model)
    elif attention != "none":
        raise ValueError(f"Unsupported attention type: {attention}")

    model.fc = classifier_head(model.fc.in_features, num_classes=num_classes)
    return model


def is_classifier_or_attention_parameter(name):
    return name.startswith("fc.") or ".1.channel_attention" in name or ".1.spatial_attention" in name
