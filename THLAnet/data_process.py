import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import process_encoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
f = open("embedding_32.txt", "r")
lines = f.readlines()
aa_dict_atchley = process_encoder.aa_dict_atch()
print(aa_dict_atchley)
for line in lines[1:21]:
    li = line.split(',')
    myarray = np.array(li[1:])
    myarray = myarray.astype(float)
    aa_dict_atchley[li[0]] = np.concatenate((aa_dict_atchley[li[0]], myarray), axis=0)

torch.set_default_dtype(torch.float64)


def pad_sequence(seq, max_len, feature_dim=21):
    """将序列填充至指定长度，填充值为 0"""
    if seq.size(1) < max_len:
        padding = torch.zeros(seq.size(0), max_len - seq.size(1), feature_dim, dtype=seq.dtype, device=seq.device)
        seq = torch.cat([seq, padding], dim=1)
    return seq


def get_TCR_antigen_result_blosum(file_path):
    """生成 TCR、抗原和 HLA 的 BLOSUM50 编码，输出形状 [n, 80, 21]"""
    TCR_list, antigen_list, HLA_seq_list = process_encoder.preprocess(file_path)

    # BLOSUM50 编码，假设 process_encoder.antigenMap 已实现填充
    antigen_array_blosum = process_encoder.antigenMap(antigen_list, 17, 'BLOSUM50')  # [n, 17, 21]
    TCR_array_blosum = process_encoder.antigenMap(TCR_list, 27, 'BLOSUM50')  # [n, 27, 21]
    HLA_array_blosum = process_encoder.antigenMap(HLA_seq_list, 36, 'BLOSUM50')  # [n, 36, 21]

    # 转换为 Tensor
    antigen_array_blosum = torch.Tensor(antigen_array_blosum)
    TCR_array_blosum = torch.Tensor(TCR_array_blosum)
    HLA_array_blosum = torch.Tensor(HLA_array_blosum)

    # 确保填充至指定长度（若 process_encoder 未填充）
    antigen_array_blosum = pad_sequence(antigen_array_blosum, 17)
    TCR_array_blosum = pad_sequence(TCR_array_blosum, 27)
    HLA_array_blosum = pad_sequence(HLA_array_blosum, 36)

    # 拼接：HLA (36) + Antigen (17) + TCR (27) = 80
    TCR_antigen_result_blosum = torch.cat((antigen_array_blosum, TCR_array_blosum,HLA_array_blosum ),
                                          dim=1)  # [n, 80, 21]

    return TCR_antigen_result_blosum


def add_position_encoding(seq):
    """为序列添加正弦位置编码，适配形状 [seq_len=80, dim=21]"""
    position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 21) for j in range(21)] for pos in range(80)])
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
    position_encoding = torch.from_numpy(position_encoding).to(seq.dtype).to(seq.device)

    padding_ids = torch.abs(seq).sum(dim=-1) == 0
    seq[~padding_ids] += position_encoding[:seq[~padding_ids].size(0)]
    return seq


def get_position_encoding_data(file_path):
    """获取带位置编码的数据，输出形状 [n, 80, 21]"""
    TCR_antigen_result_blosum = get_TCR_antigen_result_blosum(file_path)  # [n, 80, 21]
    position_input_data = torch.zeros_like(TCR_antigen_result_blosum)
    for i in tqdm(range(len(TCR_antigen_result_blosum)), desc="添加位置编码"):
        position_data = add_position_encoding(TCR_antigen_result_blosum[i])
        position_input_data[i] = position_data
    return position_input_data