import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
class BilinearMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in1, dim_in2, dim_out, num_heads):
        super(BilinearMultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out

        # 确保输出维度可以被头数整除
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        self.head_dim = dim_out // num_heads

        # 定义线性映射层
        self.linear_q = nn.Linear(dim_in1, dim_out, bias=False)
        self.linear_k = nn.Linear(dim_in2, dim_out, bias=False)
        self.linear_v = nn.Linear(dim_in1, dim_out, bias=False)

        self.fc_out = nn.Linear(dim_out, dim_out)  # 最后的线性变换

    def forward(self, x1_chunk, x2_chunk):
        # 保持所有数据为 float32
        x1_chunk, x2_chunk = x1_chunk.float(), x2_chunk.float()

        # 线性投影
        Q = self.linear_q(x1_chunk)  # [chunk_size, seq_len, dim_out]
        K = self.linear_k(x2_chunk)  # [chunk_size, seq_len, dim_out]
        V = self.linear_v(x1_chunk)  # [chunk_size, seq_len, dim_out]

        # 分头处理
        batch_size, seq_len, _ = Q.size()
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        # 计算双线性注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)  # 归一化权重

        # 加权求和
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim_out)  # [batch_size, seq_len, dim_out]

        # 最后的线性变换
        output = self.fc_out(output)

        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 参数设置
dim_in1 = 1280
dim_in2 = 21
dim_out = 1000
num_heads = 8  # 多头的数量
seq_len = 79
chunk_size = 5000  # 每次处理的样本数量

parser=argparse.ArgumentParser()
parser.add_argument('--esm_data',type=str,help='Please input esm_data set file path')
parser.add_argument('--transformer_data',type=str,help='Please input transformer_data set file path')
args=parser.parse_args()
# 创建数据

x1=torch.load(args.esm_data)
x2=torch.load(args.transformer_data)

x2=torch.Tensor(x2)
print("bilinearmultihead")
# 实例化模型并移动到 GPU
bilinear_self_attention_optimized =  BilinearMultiHeadSelfAttention(dim_in1, dim_in2, dim_out, num_heads).cuda()
# 分块处理，保存输出
output_chunks = []
for i in range(0, len(x1), chunk_size):
    # 从CPU加载当前块到GPU
    x1_chunk = x1[i:i + chunk_size].cuda()
    x2_chunk = x2[i:i + chunk_size].cuda()

    # 处理当前块并移回CPU
    with torch.no_grad():  # 关闭梯度计算，节省内存
        output_chunk = bilinear_self_attention_optimized(x1_chunk, x2_chunk)
        output_chunks.append(output_chunk.cpu())  # 移回 CPU 减少 GPU 占用

    # 释放当前块的GPU显存
    del x1_chunk, x2_chunk, output_chunk
    torch.cuda.empty_cache()

# 将所有块合并为最终输出
output = torch.cat(output_chunks, dim=0)  # [61000, 79, 1000]

torch.save(output,'esm2_transfomer_bilinerattention.pt')
print("输出形状:", output.shape)
