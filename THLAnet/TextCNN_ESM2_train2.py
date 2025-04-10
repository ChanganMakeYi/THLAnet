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
# parser.add_argument('--data_esm',type=str,help='Please input train_data_esm set file path')
parser.add_argument('--data_transfomer_esm',type=str,help='Please input train_data_esm set file path')
parser.add_argument('--model_esm_path',type=str,help='Please input model file path')
args=parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



embedding_dim = 77  # Embedding dimension
kernel_sizes = [1, 1, 1]  # Different kernel sizes for convolutionspip
num_filters = 1000  # Number of filters per kernel size


#model_pre = tc2.TextCNN(embedding_dim, kernel_sizes, num_filters)

model_pre = mlp.MLPtrain()

model_pre=model_pre.to(device)
optimizer = torch.optim.Adam(model_pre.parameters(), lr=0.00005,foreach=False)


#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
label_cancer=process_encoder.get_label(args.file_path_train)
label_cancer=torch.Tensor(label_cancer)


#
pHLA_TCR_result=torch.load(args.data_transfomer_esm)
#
# print('1-1')
# torch.cuda.empty_cache()
# print('1-1')
#
#
# pHLA_input_result=torch.cat((pHLA_TCR_result,transfomer_input_data),dim=2)
# torch.save(pHLA_input_result,'pHLA_input_result.pt')
# 从 HDF5 文件中提取数据
#with h5py.File(args.data_transfomer_esm, "r") as h5f:
    # 提取数据集 'final_output'，转换为 NumPy 数组
#    final_output_data = h5f["final_output"][:]

    # 将 NumPy 数组转换为 PyTorch Tensor
#    final_output_tensor = torch.tensor(final_output_data)

# 创建 TensorDataset
transfomer_esm_input_data = pHLA_TCR_result

# transfomer_esm_input_data=torch.load(dataset)
print(transfomer_esm_input_data.shape)
# torch.cuda.empty_cache()
print('1-2')



# 加载数据
print(1)
# TCR_antigen_result_sum=TCR_antigen_result_blosum.to(device)
# TCR_antigen_result_sum_truan=transfomer_esm_input_data[:train_size,:,:1280]
# TCR_antigen_result_sum_test=transfomer_esm_input_data[train_size:length,:,:1280]
#
# TCR_antigen_result_sum_truan=transfomer_esm_input_data[:train_size]
# TCR_antigen_result_sum_test=transfomer_esm_input_data[train_size:length]
# print(TCR_antigen_result_sum_truan.shape)
# print(TCR_antigen_result_sum_test.shape)
# # 假设 dataset 是你的数据集对象，length 是数据集的长度
label_cancer=label_cancer.to(device)
# TCR_antigen_result_sum_truan=TCR_antigen_result_sum_truan.to(device)
# dataset = TensorDataset(TCR_antigen_result_sum_truan,label_cancer[:train_size])
# 定义划分比例，比如训练集占80%，验证集占20%

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
        batch_x = torch.tensor(batch_x, dtype=torch.float32)
        batch_y= torch.tensor(batch_y, dtype=torch.float32)
        batch_y = batch_y.unsqueeze(1)
        output = model_pre(torch.tensor(batch_x.data, dtype=torch.float32))
        # print(str(epoch)+'_________________________')
        # print(batch_x.data)
        # print(output.data)
        # print('_________________________')
        # print(output)
        loss = criterion(output, batch_y)
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
        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels= torch.tensor(labels, dtype=torch.float32)
        labels = labels.unsqueeze(1)

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




