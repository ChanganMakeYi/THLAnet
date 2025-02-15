import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--file1',type=str,help='Please input train_data set file path')
parser.add_argument('--file2',type=str,help='Please input output file path')
parser.add_argument('--file3',type=str,help='Please input the embedding type')
parser.add_argument('--output',type=str,help='Please output file')
args=parser.parse_args()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
# 假设你的数据张量如下
data1 = torch.load(args.file1)  # [60000,17,1280]
data2 = torch.load(args.file2)  # [60000,27,1280]
data3 = torch.load(args.file3)  # [60000,35,1280]

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 分批大小
batch_size = 128  # 正常批次大小
# last_batch_size = 106  # 最后一个批次大小
#
# # 计算总批次数（除最后一个批次外）
total_samples = data1.size(0)
# num_batches = total_samples // batch_size
# if total_samples % batch_size != 0:
#     num_batches += 1  # 如果样本数不是 batch_size 的倍数，最后一个批次

print(data1.shape)
print(data2.shape)
print(data3.shape)
# 创建 TensorDataset 和 DataLoader
dataset = TensorDataset(data1, data2, data3)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 定义交叉注意力层
embed_dim = 1280
num_heads =2  # 假设使用8个注意力头
cross_attention1 = nn.MultiheadAttention(embed_dim, num_heads).to(device)
cross_attention2 = nn.MultiheadAttention(embed_dim, num_heads).to(device)

# 创建HDF5文件，增量保存
output_file_h5 = args.output


with h5py.File(output_file_h5, "w") as h5f:
    # 预创建数据集, 每次写入对应 batch_size 的数据
    final_output_shape = (total_samples, 17, 1280)

    dset = h5f.create_dataset("final_output", shape=final_output_shape, dtype='float32',
                              chunks=(batch_size, 17, 1280))
    # 分批处理数据并保存到 HDF5
    start_idx = 0
    for i, (batch_data1, batch_data2, batch_data3) in enumerate(dataloader):
        # 将批次数据转移到 GPU
        current_batch_size = batch_data1.size(0)

        batch_data1 = batch_data1.transpose(0, 1).to(device)  # [17, batch_size, 1280]
        batch_data2 = batch_data2.transpose(0, 1).to(device)  # [27, batch_size, 1280]
        batch_data3 = batch_data3.transpose(0, 1).to(device)  # [35, batch_size, 1280]



        attn_output1, _ = cross_attention1(batch_data1, batch_data2, batch_data2)

        final_output, _ = cross_attention2(attn_output1, batch_data3, batch_data3)

        # 转换回 [batch_size, seq_len, embed_dim] 形式并转移到 CPU
        final_output = final_output.transpose(0, 1).cpu().detach().numpy()

        # 写入 HDF5 文件
        end_idx = start_idx + current_batch_size
        # print(end_idx-start_idx)
        dset[start_idx:end_idx] = final_output
        start_idx = end_idx

        # 清除 GPU 显存
        del batch_data1, batch_data2, batch_data3, attn_output1, final_output
        torch.cuda.empty_cache()

# # 将 HDF5 文件内容加载并保存为 .pt 文件
# with h5py.File(output_file_h5, "r") as h5f:
#     final_output_tensor = torch.tensor(h5f["final_output"][:])
#     torch.save(final_output_tensor, output_file_pt)
#
# print("Final output saved to HDF5 file:", output_file_h5)