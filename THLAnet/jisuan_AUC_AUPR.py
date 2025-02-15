import csv
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


dataset = pd.read_csv('data/compare_pmtnet.csv')

Score = list(dataset['Score'].dropna())
Label = list(dataset['label'])

auc = roc_auc_score(np.array(Label), np.array(Score))
aupr = average_precision_score(np.array(Label), np.array(Score))
# accuracy = accuracy_score(np.array(Label), np.array(Score))


print(auc)
print(aupr)
# print(accuracy)