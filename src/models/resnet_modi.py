import torch
from torch import nn

from torchvision.models import resnet18


class Cifar10Resnet18(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()

        self.model = resnet18(num_classes=num_classes)

        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)
