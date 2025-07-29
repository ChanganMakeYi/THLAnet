import numpy as np
import umap
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图支持
import torch
import argparse

# 清空 GPU 缓存（如果使用 GPU）
torch.cuda.empty_cache()

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--file1', type=str, help='请输入张量数据集文件路径')
parser.add_argument('--file2', type=str, help='请输入标签数据集文件路径')
args = parser.parse_args()

# 加载张量数据
try:
    data_matrix = torch.load(args.file1)
except FileNotFoundError:
    print(f"错误：文件 {args.file1} 不存在")
    exit(1)

# 展平张量为二维
data_matrix = data_matrix.reshape(len(data_matrix), -1)
print(f"数据形状：{data_matrix.shape}")

# 加载标签数据
try:
    labels = pd.read_csv(args.file2)["label"]
except FileNotFoundError:
    print(f"错误：文件 {args.file2} 不存在")
    exit(1)
except KeyError:
    print("错误：CSV 文件中缺少 'label' 列")
    exit(1)

# 验证标签和数据点数量是否匹配
if len(labels) != len(data_matrix):
    print(f"错误：标签数量 ({len(labels)}) 与数据点数量 ({len(data_matrix)}) 不匹配")
    exit(1)

# 执行 UMAP 降维到 3 维
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', n_components=3)
umap_result = reducer.fit_transform(data_matrix)

# 转换为 DataFrame 并添加标签
umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2", "UMAP3"])
umap_df["label"] = labels

# 绘制 3D 散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 根据标签类别着色
scatter = ax.scatter(
    umap_df["UMAP1"],
    umap_df["UMAP2"],
    umap_df["UMAP3"],
    c=umap_df["label"].astype('category').cat.codes,
    cmap="Spectral",
    alpha=0.6
)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_ticks(range(len(umap_df["label"].unique())))
cbar.set_label("标签类别")

# 设置标题和轴标签
ax.set_title("3D UMAP 降维结果")
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_zlabel("UMAP3")

# 保存图像
plt.savefig("umap_3d_plot.png", dpi=300, bbox_inches='tight')
plt.close()