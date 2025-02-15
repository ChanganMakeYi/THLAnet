import torch
import torch.nn as nn
import argparse
# 定义双线性注意力网络类

class BilinearAttentionNetwork(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(BilinearAttentionNetwork, self).__init__()
        self.W1 = nn.Parameter(torch.randn(feature_dim, feature_dim))  # T-H 权重
        self.W2 = nn.Parameter(torch.randn(feature_dim, feature_dim))  # H-A 权重
        self.V = nn.Parameter(torch.randn(feature_dim, output_dim))  # 输出映射权重
        self.activation = nn.ReLU()

    def forward(self, T, H, A):
        # 修正后的 T 和 H 交互
        T_H_interaction = torch.einsum('bij,jk,bik->bi', T, self.W1, H.transpose(1, 2))  # [70000, 1280]

        # 修正后的 H 和 A 交互
        H_A_interaction = torch.einsum('bij,jk,bik->bi', H, self.W2, A.transpose(1, 2))  # [70000, 1280]

        # 融合特征
        fused_feature = T_H_interaction + H_A_interaction  # [70000, 1280]

        # 映射到输出维度
        output = fused_feature @ self.V  # [70000, output_dim]
        output = self.activation(output)

        return output

parser=argparse.ArgumentParser()
parser.add_argument('--file1',type=str,help='Please input train_data set file path')
parser.add_argument('--file2',type=str,help='Please input train_data set file path')
parser.add_argument('--file3',type=str,help='Please input train_data_esm set file path')
args=parser.parse_args()

# 数据准备
T = torch.load(args.file1)  # 张量1
H = torch.load(args.file2) # 张量2
A = torch.load(args.file3)  # 张量3

# 定义BAN
feature_dim = 1280  # 输入特征维度
output_dim = 1024   # 输出特征维度 (你可以根据需求调整)
ban = BilinearAttentionNetwork(feature_dim, output_dim)

# 前向传播
fused_features = ban(T, H, A)

print("Fused Feature Shape:", fused_features.shape)  # 输出形状: [70000, 1024]
torch.save(fused_features,'ban_feature.pt')