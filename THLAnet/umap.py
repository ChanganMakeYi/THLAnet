
# 导入库
import numpy as np
import umap
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
torch.cuda.empty_cache()
parser=argparse.ArgumentParser()
parser.add_argument('--file1',type=str,help='Please input test_data set file path')
parser.add_argument('--file2',type=str,help='Please input test_data set file path')
args=parser.parse_args()

data_matrix=np.array(torch.load(args.file))

# 将 Tensor 转换为 NumPy 数组并展平成二维
#data_matrix = data_matrix.view(66314, -1).numpy()  # shape (60000, 10000)
data_matrix=data_matrix.reshape(len(data_matrix),-1)

print(data_matrix.shape)


# 加载标签数据，假设标签在CSV文件中，列名为 "label"
labels = pd.read_csv(args.file2)["label"]

# 执行 UMAP 降维
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
umap_result = reducer.fit_transform(data_matrix)

# 将结果转换为 DataFrame，并添加标签列
umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"])
umap_df["label"] = labels
# 绘图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(umap_df["UMAP1"], umap_df["UMAP2"], c=umap_df["label"].astype('category').cat.codes, cmap="Spectral", alpha=0.6)
plt.colorbar(scatter, ticks=range(len(umap_df["label"].unique())))
plt.title("UMAP Dimensionality Reduction with Labels")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")

plt.savefig("umap_plot.png", dpi=300, bbox_inches='tight')