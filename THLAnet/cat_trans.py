import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import argparse

# parser=argparse.ArgumentParser()
# parser.add_argument('--file_path_posit',type=str,help='Please input train_data set file path')
# args=parser.parse_args()

transfomer_input_result=np.concatenate((torch.load('transfomer_cold_test1.pt'),torch.load('transfomer_cold_test2.pt')),axis=0)
transfomer_input_result=torch.Tensor(transfomer_input_result)
print(transfomer_input_result.shape)

# 前向传播
torch.save(transfomer_input_result,'transfomer_cold_test.pt')

# transfomer_data=torch.load('transfomer_input_result.pt')
# transfomer_data=torch.Tensor(transfomer_data)
# esm2_data=torch.load('pHLA_TCR_result.pt')
#
# esm2_transfomer_data=torch.cat((esm2_data,transfomer_data),dim=2)
#
# torch.save(esm2_transfomer_data,'esm2_transfomer_data.pt')
# print(esm2_transfomer_data.shape)
