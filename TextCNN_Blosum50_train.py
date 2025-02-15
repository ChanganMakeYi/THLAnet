from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
import data_process
import process_encoder
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import TextCNN as tc
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import argparse
import os as os

parser=argparse.ArgumentParser()
parser.add_argument('--file_path_train',type=str,help='Please input train_data set file path')
parser.add_argument('--transfomer_data_path',type=str,help='Please input train_data set file path')
parser.add_argument('--epoch',type=int,help='Please input train_data set file path')
parser.add_argument('--model_esm_path',type=str,help='Please input model file path')
args=parser.parse_args()

embedding_dim = 79  # Embedding dimension
kernel_sizes = [1, 2, 1, 2]  # Different kernel sizes for convolutionspip
num_filters = 1000  # Number of filters per kernel size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


label_cancer=process_encoder.get_label(args.file_path_train)
label_cancer=torch.Tensor(label_cancer)
label_cancer=label_cancer.to(device)
model_pre = tc.TextCNN(embedding_dim, kernel_sizes, num_filters)
# 实例化模型
# position_input_data_cancer=0
# if os.path.exists('position_input_data_cancer.npy'):
#     position_input_data_cancer=np.load('position_input_data_cancer.npy')
# else:
#     position_input_data_cancer=data_process.get_position_encoding_data(args.file_path_train)
#     np.save('position_input_data_cancer.npy',position_input_data_cancer)




model_pre=model_pre.to(device)

criterion = nn.CrossEntropyLoss()

transfomer_data=torch.load(args.transfomer_data_path)

transfomer_data=torch.Tensor(transfomer_data)
transfomer_data=transfomer_data.to(device)


optimizer = torch.optim.Adam(model_pre.parameters(), lr=0.00005,foreach=False)

# 加载数据

# TCR_antigen_result_sum=TCR_antigen_result_blosum.to(device)
TCR_antigen_result_sum=transfomer_data.to(device)

# 假设 dataset 是你的数据集对象，length 是数据集的长度
dataset = TensorDataset(TCR_antigen_result_sum,label_cancer)
length = len(dataset)

# 定义划分比例，比如训练集占80%，验证集占20%
train_size = int(0.8 * length)
val_size = length - train_size

# 使用 random_split 函数划分数据集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloader = DataLoader(train_dataset, batch_size=256)

loss_sum = np.zeros(args.epoch)
# 训练模型
for epoch in tqdm(range(args.epoch)):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model_pre(batch_x.data)
        loss = criterion(output, batch_y.long())
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    loss_sum[epoch]=loss.item()




torch.save(model_pre.state_dict(), args.model_esm_path)




test_dataloader = DataLoader(val_dataset, batch_size=256)

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
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        try:
            i += 1
            auc += roc_auc_score(np.array(labels.cpu()), np.array(predicted.cpu()))
            aupr += average_precision_score(np.array(labels.cpu()), np.array(predicted.cpu()))
            accuracy += accuracy_score(np.array(labels.cpu()), np.array(predicted.cpu()))
        #             prAUC+=pr_auc(np.array·(labels.cpu()), np.array(predicted.cpu()))
        except ValueError:
            print(111)
            pass
    auc_sum = auc / i
    aupr_sum = aupr / i
    Accuracy = accuracy / i
    print('AUC:', auc_sum)
    print('AUPR:', aupr_sum)
    print("Accuracy:", Accuracy)