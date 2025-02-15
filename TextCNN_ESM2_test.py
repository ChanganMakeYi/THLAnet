import pandas as pd

import esm
import TextCNN2 as tc2

from sklearn.metrics import roc_auc_score, average_precision_score

import process_encoder
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import argparse
import data_process

parser=argparse.ArgumentParser()
parser.add_argument('--file_path_train',type=str,help='Please input train_data set file path')
parser.add_argument('--data_transfomer_esm',type=str,help='Please input train_data_esm set file path')
parser.add_argument('--model_esm_path',type=str,help='Please input model file path')
args=parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transfomer_input_data=data_process.get_position_encoding_data(args.file_path_train)


# transfomer_input_data=transfomer_input_data.reshape(-1,79,21)
# torch.save(transfomer_input_data,'transfomer_input_data.pt')

# transfomer_input_data=torch.load(args.data_transfomer_esm)
embedding_dim = 79  # Embedding dimension
kernel_sizes = [1, 2, 3, 2]  # Different kernel sizes for convolutionspip
num_filters = 1000  # Number of filters per kernel size


model_pre = tc2.TextCNN(embedding_dim, kernel_sizes, num_filters)


model_pre.load_state_dict(torch.load(args.model_esm_path))
model_pre=model_pre.to(device)

criterion = nn.CrossEntropyLoss()
label_cancer=process_encoder.get_label(args.file_path_train)
label_cancer=torch.Tensor(label_cancer)



transfomer_esm_input_data=torch.load(args.data_transfomer_esm)

length = len(label_cancer)
train_size = int(0.8 * length)
val_size=length-train_size


TCR_antigen_result_sum_test=transfomer_esm_input_data[train_size:length,:,:1280]




TCR_antigen_result_sum_test=TCR_antigen_result_sum_test.to(device)
print(TCR_antigen_result_sum_test.shape)
dataset_test = TensorDataset(TCR_antigen_result_sum_test,label_cancer[train_size:length])
test_dataloader = DataLoader(dataset_test, batch_size=256,shuffle=True)

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
        inputs = torch.tensor(inputs, dtype=torch.double)
        outputs = model_pre(inputs.data)
        _, predicted = torch.max(outputs.data, 1)
        predicted=predicted.to(device)
        labels=labels.to(device)
        correct += (predicted == labels).sum().item()
        print(((predicted == labels).sum().item())/256)
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
    print(correct / len(TCR_antigen_result_sum_test))
    auc_sum = auc / i
    aupr_sum = aupr / i
    Accuracy = accuracy / i
    print('AUC:', auc_sum)
    print('AUPR:', aupr_sum)
    print("Accuracy:", Accuracy)