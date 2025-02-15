
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
import data_process
import process_encoder
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import TextCNN2 as tc

import pandas as pd

import esm
import data_process

import torch
import argparse
import TransformerEncoderLayer

torch.cuda.empty_cache()
parser=argparse.ArgumentParser()
parser.add_argument('--file_path_test',type=str,help='Please input test_data set file path')
parser.add_argument('--model_path',type=str,help='Please input test_data set file path')
args=parser.parse_args()

print(1)
file_path=args.file_path_test
#
print(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(1)
dataset = pd.read_csv(file_path)
print(1)
transfomer_input_data=data_process.get_position_encoding_data(file_path)
print(2)
transfomer_result_data=TransformerEncoderLayer.transfomers_encoder_batch(transfomer_input_data)
print(3)

transfomer_result_data=(torch.Tensor(transfomer_result_data)).to(device)


Antigen=list(dataset['Antigen'].dropna())
CDR3=list(dataset['CDR3'].dropna())
HLA=list(dataset['HLA_seq'].dropna())




print(4)

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model=model.to(device)
batch_converter = alphabet.get_batch_converter()

model.eval()  # disables dropout for deterministic results
print(5)
# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)

print(2)
Antigen_list = []
CDR3_list=[]
HLA_list=[]
for i in range(len(CDR3)):
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
# result_attentions = (torch.Tensor(10, 33, 20, 27, 27))
# result_representations=(torch.Tensor(10, 27,1280))
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



embedding_dim = 79  # Embedding dimension
kernel_sizes = [1, 2, 3, 2]  # Different kernel sizes for convolutionspip
num_filters = 1000  # Number of filters per kernel size

model_pre = tc.TextCNN(embedding_dim, kernel_sizes, num_filters)

model_pre.load_state_dict(torch.load(args.model_path,map_location=torch.device('cpu')))

model_pre=model_pre.to(device)

pHLA_TCR_result=pHLA_TCR_result.to(device)
label_cancer=process_encoder.get_label(file_path)
esm2_transfomer_data=torch.cat((pHLA_TCR_result,transfomer_result_data[:len(pHLA_TCR_result)]),dim=2)

label_cancer=torch.Tensor(label_cancer[:len(pHLA_TCR_result)])
dataset = TensorDataset(esm2_transfomer_data, label_cancer)

test_dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# model_pre.eval()
test_loss = 0
correct = 0
total = 0
r2 = 0

# 加载保存的模型参数

with torch.no_grad():
    model_pre.eval()
    correct = 0
    total = 0
    auc = 0
    aupr = 0
    accuracy = 0
    i = 0
    prAUC = 0
    for inputs, labels in tqdm(test_dataloader):

        outputs = model_pre(inputs.data)
        test1, predicted = torch.max(outputs.data, 1)
        predicted=predicted.to(device)
        print(test1)
        for i in range(len(test1)):
            print(test1[i])
        labels=labels.to(device)
        correct += (predicted == labels).sum().item()
        print(((predicted == labels).sum().item())/len(inputs))
        total += labels.size(0)
        try:
            print("auc")
            auc += roc_auc_score(np.array(labels.cpu()), np.array(predicted.cpu()))
            print("aupr")
            aupr += average_precision_score(np.array(labels.cpu()), np.array(predicted.cpu()))
            print('auc:'+str(roc_auc_score(np.array(labels.cpu()), np.array(predicted.cpu()))),'aupr:'+ str(average_precision_score(np.array(labels.cpu()), np.array(predicted.cpu()))))
            i += 1
        except ValueError:
            print(111)
            pass
    print(i)
    print(correct / len(pHLA_TCR_result))
    auc_sum = auc / i
    aupr_sum = aupr / i
    Accuracy = accuracy / i
    print('AUC:', auc_sum)
    print('AUPR:', aupr_sum)
    print("Accuracy:", Accuracy)


