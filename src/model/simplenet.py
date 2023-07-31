import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 32 * 32, 10),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x.flatten())
        return x


# if __name__ == "__main__":
#     net = SimpleNet()
#     img = torch.zeros((1, 3, 32, 32))
#     out = net(img)
#     print(out.shape)
