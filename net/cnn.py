import torch.nn as nn
import torch.nn.functional as F

## byol
class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=512):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
       
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
       

    def forward(self, x):
      
          x = self.layer1(x)
          x = self.layer2(x)
       
          return x

class CNN(nn.Module):
    def __init__(self, channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)
        self.pro = projection_MLP(32)

    def forward(self, x, return_projection = False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # forward
        if return_projection:
            # global pooling
            x = self.global_avg_pool(x)
            x = x.squeeze(-1)
            # projection
            x = self.pro(x)
            return x
        else:
            x = self.global_avg_pool(x)
            x = x.squeeze(-1)
            x = self.fc(x)
            return x