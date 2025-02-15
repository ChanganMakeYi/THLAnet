
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
import data_process
import process_encoder
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import TextCNN as tc
import argparse

torch.cuda.empty_cache()
parser=argparse.ArgumentParser()
parser.add_argument('--file_path_test',type=str,help='Please input test_data set file path')
parser.add_argument('--model_path',type=str,help='Please input model file path')
args=parser.parse_args()



embedding_dim = 79  # Embedding dimension
kernel_sizes = [1, 2, 1, 2]  # Different kernel sizes for convolutionspip
num_filters = 100  # Number of filters per kernel size




model_pre = tc.TextCNN(embedding_dim, kernel_sizes, num_filters)

model_pre.load_state_dict(torch.load(args.model_path,map_location='cpu'))

file_path=args.file_path_test
position_input_data_cancer=data_process.get_position_encoding_data(file_path)
label_cancer=process_encoder.get_label(file_path)

# label_cancer = np.ones(len(position_input_data_cancer))



position_input_data_cancer=torch.Tensor(position_input_data_cancer)
label_cancer=torch.Tensor(label_cancer)
dataset = TensorDataset(position_input_data_cancer, label_cancer)

test_dataloader = DataLoader(dataset, batch_size=256)

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
    for inputs, labels in test_dataloader:

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
    print(correct)
    print(len(position_input_data_cancer))
    print(i)
    print(correct / len(position_input_data_cancer))
    auc_sum = auc / i
    aupr_sum = aupr / i
    #     prAUC_sum=prAUC/19
    Accuracy = accuracy / i
    print('AUC:', auc_sum)
    print('AUPR:', aupr_sum)
    print("Accuracy:", Accuracy)