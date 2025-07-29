import numpy as np
import torch
import argparse


torch.cuda.empty_cache()
parser=argparse.ArgumentParser()
parser.add_argument('--file1',type=str,help='Please input test_data set file path')
parser.add_argument('--file2',type=str,help='Please input test_data set file path')
parser.add_argument('--file3',type=str,help='Please input test_data set file path')
args=parser.parse_args()
# 文件路径
file1_path = args.file1
file2_path = args.file2
file3_path = args.file3
output_path = 'output.pt'

# 加载 .npy 文件
array1 = np.load(file1_path)
array2 = np.load(file2_path)
array3 = np.load(file3_path)

# 检查数组形状（确保第一个和第三个维度一致）
assert array1.shape[0] == array2.shape[0] == array3.shape[0], "第一个维度不匹配"
assert array1.shape[2] == array2.shape[2] == array3.shape[2], "第三个维度不匹配"

# 沿第二个维度拼接
concatenated_array = np.concatenate([array1, array2, array3], axis=1)

# 转换为 PyTorch 张量
tensor = torch.from_numpy(concatenated_array)

# 保存为 .pt 文件
torch.save(tensor, output_path)

print(f"成功拼接并保存为 {output_path}，形状为 {tensor.shape}")