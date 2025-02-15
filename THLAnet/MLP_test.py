
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
import data_process
import process_encoder
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import TextCNN2 as tc
import MLPtrain as mlp
import pandas as pd
from tqdm import tqdm
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


model_pre = mlp.MLPtrain()

model_pre.load_state_dict(torch.load(args.model_path,map_location=torch.device('cpu')))

model_pre=model_pre.to(device)

label_cancer=process_encoder.get_label(file_path)
label_cancer=torch.Tensor(label_cancer)
transfomer_input_data=transfomer_input_data.to(device)
label_cancer=label_cancer.to(device)
dataset = TensorDataset(transfomer_input_data, label_cancer)

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
    auc_sum = auc / i
    aupr_sum = aupr / i
    Accuracy = accuracy / i
    print('AUC:', auc_sum/i)
    print('AUPR:', aupr_sum/i)
    print("Accuracy:", Accuracy)


