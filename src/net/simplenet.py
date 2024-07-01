import torch
from torch import nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=35, stride=16, hidden_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * hidden_dim)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * hidden_dim, 2 * hidden_dim, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * hidden_dim)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2).squeeze(1)


if __name__ == "__main__":
    net = SimpleNet()
    inp = torch.randn((2, 1, 16000))
    out = net(inp)
    print(out.shape)
