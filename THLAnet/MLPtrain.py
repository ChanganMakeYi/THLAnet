import torch.nn.functional as F
import torch
import torch.nn as nn

class MLPtrain(nn.Module):
    def __init__(self):
        super(MLPtrain, self).__init__()

        # self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, kernel_size) for kernel_size in kernel_sizes])
        # 池化层

        # 全连接层
        self.fc1 = nn.Linear(79*1280, 2000)
        self.bn1 = nn.LayerNorm(2000)
        self.fc2 = nn.Linear(2000, 1048)
        self.bn2 = nn.LayerNorm(1048)
        self.fc3 = nn.Linear(1048, 2)
        self.bn3 = nn.LayerNorm(2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):

        x=x.view(-1,79*1280)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        return F.softmax(x, dim=1)

