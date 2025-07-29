import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLPBlock(nn.Module):
    """残差 MLP 块，包含全连接层和层归一化。"""
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super(ResidualMLPBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

        # Kaiming 初始化
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.kaiming_normal_(self.shortcut.weight, mode='fan_in', nonlinearity='relu')
            if self.shortcut.bias is not None:
                nn.init.zeros_(self.shortcut.bias)

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.ln(self.fc(x)))
        x = self.dropout(x)
        x = x + residual
        return F.relu(x)

class MLPtrain(nn.Module):
    """优化的 MLP 模型，输入形状 [batch_size, 80, 21]，输出 logits。

    参数：
        hidden_dims (list)：隐藏层维度列表。
        dropout_rate (float)：用于正则化的丢弃率。

    属性：
        layers (nn.ModuleList)：残差 MLP 块列表。
        fc_out (nn.Linear)：输出层。
        dropout (nn.Dropout)：正则化丢弃层。
    """
    def __init__(self, hidden_dims=[1024, 512, 256], dropout_rate=0.3):
        super(MLPtrain, self).__init__()
        input_dim = 80 * 1280
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(ResidualMLPBlock(prev_dim, dim, dropout_rate))
            prev_dim = dim
        self.layers = nn.ModuleList(layers)
        self.fc_out = nn.Linear(prev_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

        # Kaiming 初始化
        nn.init.kaiming_normal_(self.fc_out.weight, mode='fan_in', nonlinearity='relu')
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """模型的前向传播。

        参数：
            x (torch.Tensor)：输入张量，形状为 [batch_size, 80, 21]。

        返回：
            torch.Tensor：输出张量，形状为 [batch_size, 1]，表示 logits。
        """
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x