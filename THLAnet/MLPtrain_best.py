import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLPBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super(ResidualMLPBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.ln(self.fc(x)))
        x = self.dropout(x)
        x = x + residual
        return F.relu(x)

class MLPtrain(nn.Module):
    def __init__(self, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super(MLPtrain, self).__init__()
        input_dim = 80 * 21
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(ResidualMLPBlock(prev_dim, dim, dropout_rate))
            prev_dim = dim
        self.layers = nn.ModuleList(layers)
        self.fc_out = nn.Linear(prev_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        return x