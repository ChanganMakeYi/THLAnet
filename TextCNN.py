import torch.nn.functional as F
import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, embedding_dim, kernel_sizes, num_filters):
        super(TextCNN, self).__init__()

        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, kernel_size) for kernel_size in kernel_sizes])
        # 池化层

        # 全连接层
        self.fc1 = nn.Linear(4000, 2000)
        self.bn1 = nn.BatchNorm1d(2000)
        self.fc2 = nn.Linear(2000, 1048)
        self.bn2 = nn.BatchNorm1d(1048)
        self.fc3 = nn.Linear(1048, 2)
        self.bn3 = nn.BatchNorm1d(2)
        # Dropout层
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        pooled_outputs = []
        x = x.reshape(-1, 74, 21)

        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            pooled_outputs.append(pooled)
        #         print(pooled_outputs)
        x = torch.cat(pooled_outputs, dim=1)
        #         print(x.shape)

        #         print(x.shape)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        return F.softmax(x, dim=1)


