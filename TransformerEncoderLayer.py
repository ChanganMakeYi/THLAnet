import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import data_process
import argparse
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.feedforward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src



parser=argparse.ArgumentParser()
parser.add_argument('--file_path_posit',type=str,help='Please input train_data set file path')
args=parser.parse_args()
#
transfomer_input_data=data_process.get_position_encoding_data(args.file_path_posit)
# print(transfomer_input_data.shape)
#pre-train with transfomer-encoder

num_layers = 3
d_model = 21
nhead = 7
dim_feedforward = 2048
dropout = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder = TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
encoder=encoder.to(device)
# encoder=encoder
# input_data=input_data
def transfomers_encoder_batch(input_data):
    return_data = torch.Tensor(np.zeros((input_data.size(0), 77, 21)))
    return_data=return_data.cpu()
    for i in tqdm(range(input_data.size(0))):
        output = encoder((input_data[i].reshape(77,21)).to(device))
        torch.cuda.empty_cache()
        return_data[i]=output.cpu()
    return_data=return_data.detach().numpy()
    return return_data




# transfomer_output_data1=transfomers_encoder_batch(transfomer_input_data[:20000])
#
# torch.save(transfomer_output_data1,'transfomer_output_data1.pt')
# del transfomer_output_data1
# transfomer_output_data2=transfomers_encoder_batch(transfomer_input_data[20000:40000])
#
# torch.save(transfomer_output_data2,'transfomer_output_data2.pt')
# del transfomer_output_data2
# transfomer_output_data3=transfomers_encoder_batch(transfomer_input_data[40000:60000])
#
# torch.save(transfomer_output_data3,'transfomer_output_data3.pt')
# del transfomer_output_data3
# transfomer_output_data4=transfomers_encoder_batch(transfomer_input_data[60000:])
#
# torch.save(transfomer_output_data4,'transfomer_output_data4.pt')
# del transfomer_output_data4

# transfomer_output_data1=transfomers_encoder_batch(transfomer_input_data)
#
# torch.save(transfomer_output_data1,'transfomer_cold_train.pt')
