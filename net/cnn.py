import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv6 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)

        return x
