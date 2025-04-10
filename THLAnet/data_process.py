import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import process_encoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
f = open("embedding_32.txt", "r")
lines = f.readlines()  # 读取全部内容
aa_dict_atchley = process_encoder.aa_dict_atch()
print(aa_dict_atchley)
for line in lines[1:21]:
    li = line.split(',')
    myarray = np.array(li[1:])
    myarray = myarray.astype(float)
    aa_dict_atchley[li[0]] = np.concatenate((aa_dict_atchley[li[0]], myarray), axis=0)

torch.set_default_dtype(torch.float64)
#antigen_array 抗原数据
def get_TCR_antigen_result_blosum(file_path):
    TCR_list,antigen_list,HLA_seq_list=process_encoder.preprocess(file_path)
    antigen_array_blosum=process_encoder.antigenMap(antigen_list,17,'BLOSUM50')
    TCR_array_blosum=process_encoder.antigenMap(TCR_list,27,'BLOSUM50')
    HLA_array_blosum=process_encoder.antigenMap(HLA_seq_list,35,'BLOSUM50')

    antigen_array_blosum=torch.Tensor(antigen_array_blosum)
    TCR_array_blosum=torch.Tensor(TCR_array_blosum)
    HLA_array_blosum=torch.Tensor(HLA_array_blosum)

    TCR_antigen_result_blosum_ori=torch.cat((HLA_array_blosum,antigen_array_blosum,TCR_array_blosum),dim=1) #TCR序列最长为24
    TCR_antigen_result_blosum=TCR_antigen_result_blosum_ori.reshape(len(antigen_list),21,-1)

    return TCR_antigen_result_blosum

def add_position_encoding(seq):
    # Sinusoidal position encoding
    position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 74) for j in range(79)] for pos in range(21)])
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
    position_encoding = torch.from_numpy(position_encoding)

    padding_ids = torch.abs(seq).sum(dim=-1) == 0
    seq[~padding_ids] += position_encoding[:seq[~padding_ids].size()[-2]]
    return seq

def get_position_encoding_data(file_path):
    TCR_antigen_result_blosum=get_TCR_antigen_result_blosum(file_path)
    # TCR_antigen_result_blosum=TCR_antigen_result_blosum.to(device)
    position_input_data = torch.zeros(TCR_antigen_result_blosum.shape[0], 21, 79)
    # position_input_data=position_input_data.to(device)
    for i in tqdm(range(len(TCR_antigen_result_blosum))):
        position_data=add_position_encoding((TCR_antigen_result_blosum[i].cpu()).reshape(21, 79))
        position_input_data[i]=position_data
    return position_input_data


