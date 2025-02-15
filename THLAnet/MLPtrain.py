import torch.nn.functional as F
import torch
import torch.nn as nn

class MLPtrain(nn.Module):
    def __init__(self):
        super(MLPtrain, self).__init__()

        # self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, kernel_size) for kernel_size in kernel_sizes])
        # 池化层

        # 全连接层
        self.fc1 = nn.Linear(77*1280, 2000).float()
        self.bn1 = nn.BatchNorm1d(2000).float()
        self.fc2 = nn.Linear(2000, 512).float()
        self.bn2 = nn.BatchNorm1d(512).float()
        self.fc3 = nn.Linear(512, 1).float()
        self.bn3 = nn.BatchNorm1d(1).float()
        # Dropout层
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # pooled_outputs = []
        # x = x.reshape(-1, 79, 1280)
        # for conv in self.convs:
        #     conv_out = F.relu(conv(x))
        #     pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
        #     pooled_outputs.append(pooled)

        # x = torch.cat(pooled_outputs, dim=1)
        x=x.view(-1,77*1280)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        #return F.softmax(x, dim=1)
        return self.bn3(x)

