
import esm

import pandas as pd


import torch

import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--file_path_train',type=str,help='Please input train_data set file path')
args=parser.parse_args()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv(args.file_path_train)

Antigen=list(dataset['Antigen'].dropna())
CDR3=list(dataset['CDR3'].dropna())
HLA=list(dataset['HLA_seq'].dropna())


print(1)
# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model=model.to(device)
batch_converter = alphabet.get_batch_converter()

model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)

print(2)
Antigen_list = []
CDR3_list=[]
HLA_list=[]
for i in range(len(Antigen)):
    newdata1=("protein"+str(i),Antigen[i])
    newdata2=("protein"+str(i),CDR3[i])
    newdata3=("protein"+str(i),HLA[i])
    Antigen_list.append(newdata1)
    CDR3_list.append(newdata2)
    HLA_list.append(newdata3)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_labels1, batch_strs1, batch_tokens1 = batch_converter(Antigen_list)
batch_labels2, batch_strs2, batch_tokens2 = batch_converter(CDR3_list)
batch_labels3, batch_str3, batch_tokens3 = batch_converter(HLA_list)
print(3)
batch_size = 50
num_batches = len(batch_tokens1) // batch_size + (len(HLA_list) % batch_size > 0)

from tqdm import tqdm
model=model.to(device)
print(4)
pHLA_TCR_result=torch.zeros((len(Antigen_list), 79, 1280))
for i in tqdm(range(num_batches)):
    start_index = i * batch_size
    if i==num_batches:
        end_index=len(Antigen_list)
    else:
        end_index = (i + 1) * batch_size
    batch_data1 = batch_tokens1[start_index:end_index].to(device)
    batch_data2 = batch_tokens2[start_index:end_index].to(device)
    batch_data3 = batch_tokens3[start_index:end_index].to(device)
    with torch.no_grad():
        results1 = model(batch_data1, repr_layers=[33], return_contacts=True)
        results2 = model(batch_data2, repr_layers=[33], return_contacts=True)
        results3 = model(batch_data3, repr_layers=[33], return_contacts=True)
        pHLA_TCR_result[start_index:end_index]=torch.cat((results1["representations"][33],results2["representations"][33],results3["representations"][33]),dim=1)
torch.save(pHLA_TCR_result,'pHLA_TCR_result.pt')

