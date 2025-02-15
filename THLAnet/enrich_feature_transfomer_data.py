import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(21, 128)   # 第一层：21 -> 128
        self.fc2 = nn.Linear(128, 256)   # 第二层：128 -> 256
        self.fc3 = nn.Linear(256, 500)   # 第三层：256 -> 500
        self.relu = nn.ReLU()             # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))       # 第一层
        x = self.relu(self.fc2(x))       # 第二层
        x = self.fc3(x)                   # 第三层
        return x.view(-1, 79, 500)        # 调整输出形状为 [66314, 79, 500]

model = ThreeLayerMLP().to(device)

transfomer_result=(torch.Tensor(torch.load('transfomer_result.pt'))).to(device)


# 变形输入数据为 (66314 * 79, 21)
input_data = transfomer_result.view(-1, 21)

# 前向传播
output = model(input_data)
print(output.shape)
torch.save(output,'transfomer_input_result.pt')

esm2_data=torch.load('pHLA_TCR_result.pt')

esm2_transfomer_data=torch.cat((esm2_data,output.cpu()),dim=2)

torch.save(esm2_transfomer_data,'esm2_transfomer500_data.pt')
print(esm2_transfomer_data.shape)