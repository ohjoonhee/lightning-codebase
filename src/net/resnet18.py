import torch
from torch import nn

from torchvision.models import resnet18


class Cifar10Resnet18(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()

        self.backbone = resnet18(num_classes=num_classes)

        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.backbone.maxpool = nn.Identity()

    def forward(self, x):
        return self.backbone(x)
