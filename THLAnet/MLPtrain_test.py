import pandas as pd

import esm
import TextCNN2 as tc2
import MLPtrain as mlp
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
import h5py
parser=argparse.ArgumentParser()
parser.add_argument('--file_path_train',type=str,help='Please input train_data set file path')
parser.add_argument('--epoch',type=int,help='Please input train_data set file path')
parser.add_argument('--data_transfomer_esm',type=str,help='Please input train_data_esm set file path')
parser.add_argument('--model_esm_path',type=str,help='Please input model file path')
args=parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model_pre = mlp.MLPtrain()

model_pre=model_pre.to(device)
optimizer = torch.optim.Adam(model_pre.parameters(), lr=0.00005,foreach=False)


criterion = nn.CrossEntropyLoss()
label_cancer=process_encoder.get_label(args.file_path_train)
label_cancer=torch.Tensor(label_cancer)


#
transfomer_esm_input_data=torch.load(args.data_transfomer_esm)
#
print('1-2')



# 加载数据
print(1)

label_cancer=label_cancer.to(device)


transfomer_esm_input_data=transfomer_esm_input_data.to(device)
# 假设 dataset 是你的数据集对象，length 是数据集的长度
dataset = TensorDataset(transfomer_esm_input_data,label_cancer)
length = len(dataset)

# 定义划分比例，比如训练集占80%，验证集占20%
train_size = int(0.8 * length)
val_size = length - train_size

# 使用 random_split 函数划分数据集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(len(train_dataset),len(val_dataset))
dataloader = DataLoader(train_dataset, batch_size=128,shuffle=True)



print(3)


loss_sum = np.zeros(args.epoch)

# 训练模型
for epoch in tqdm(range(args.epoch)):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        batch_x = torch.tensor(batch_x, dtype=torch.double)
        output = model_pre(batch_x.data)
        loss = criterion(output, batch_y.long())
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    loss_sum[epoch]=loss.item()


print(4)


torch.save(model_pre.state_dict(), args.model_esm_path)



torch.cuda.empty_cache()




test_dataloader = DataLoader(val_dataset, batch_size=128)

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
        inputs = torch.tensor(inputs, dtype=torch.float64)

        outputs = model_pre(inputs.data)
        print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)
        predicted=predicted.to(device)
        print(predicted)
        labels=labels.to(device)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        try:
            auc += roc_auc_score(np.array(labels.cpu()), np.array(predicted.cpu()))
            aupr += average_precision_score(np.array(labels.cpu()), np.array(predicted.cpu()))
            print('auc:'+str(roc_auc_score(np.array(labels.cpu()), np.array(predicted.cpu()))),'aupr:'+ str(average_precision_score(np.array(labels.cpu()), np.array(predicted.cpu()))))
            i += 1
        except ValueError:
            print(111)
            pass
    print(i)
    # print(correct / len(TCR_antigen_result_sum_test))
    auc_sum = auc / i
    aupr_sum = aupr / i
    Accuracy = accuracy / i
    auc_and_aupr=str(auc_sum)+'--'+str(aupr_sum)
    print('AUC:', auc_sum)
    print('AUPR:', aupr_sum)
    # 使用 'a' 模式打开文件（追加模式）
    with open("data/output.txt", "a", encoding="utf-8") as file:
        file.write(auc_and_aupr)




